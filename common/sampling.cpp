#include <ctime>
#include <map>
#include <cstring>
#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <unordered_map>
#include <vector>

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    std::mt19937 rng_spec; // rejection-sampling verification draws

    // backend sampling was requested: usable while no grammar constrains the tokens (common_sampler_backend_ok)
    bool backend_wanted = false;

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }

    void set_logits(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
        } else {
            const auto * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) ||
        params.penalty_repeat <= 0.0f ||
        !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) {
        throw std::invalid_argument("penalty_freq must be finite");
    }
    if (!std::isfinite(params.penalty_present)) {
        throw std::invalid_argument("penalty_present must be finite");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    // logit bias: user biases + model suppress tokens (-INFINITY)
    {
        std::vector<llama_logit_bias> merged = params.logit_bias;

        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            merged.push_back({ suppress[i], -INFINITY });
        }

        if (!merged.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged.size(), merged.data()));
        }
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // default: sample from distribution
            samplers.push_back(llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    // a grammar that constrains the tokens needs the full logits on the host. A lazy grammar constrains nothing
    // until its trigger: callers that re-check common_sampler_backend_ok() before each decode (the server) can
    // sample on the backend until then. The reasoning budget only forces tokens, which common_sampler_sample
    // applies on top of a backend-sampled token, so it keeps backend sampling.
    const bool backend_wanted = params.backend_sampling;

    if (grmr && params.backend_sampling) {
        if (!params.grammar_lazy) {
            LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);
        }

        params.backend_sampling = false;
    }

    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
        /* .rng_spec = */ std::mt19937(llama_sampler_get_seed(chain) ^ 0x9e3779b9u),
        /* .backend_wanted = */ backend_wanted,
    };

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

static bool grammar_should_apply(const struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

// the grammar (if any) constrains the next token; a lazy grammar does not before its trigger
static bool grammar_constrains(const struct common_sampler * gsmpl) {
    return grammar_should_apply(gsmpl) && !llama_sampler_grammar_awaiting_trigger(gsmpl->grmr);
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);

        // if done, replay end sequence which may contain a grammar trigger
        const bool is_done = common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_DONE;
        if (gsmpl->grmr && !accept_grammar && is_done) {
            const llama_tokens * end_seq = common_reasoning_budget_get_end_match(gsmpl->rbudget);
            if (end_seq) {
                for (const llama_token end_token : *end_seq) {
                    llama_sampler_accept(gsmpl->grmr, end_token);
                }
            }
        }
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    return new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
        /* .rng_spec = */ gsmpl->rng_spec,
        /* .backend_wanted = */ gsmpl->backend_wanted,
    };
}

void common_sampler_copy(const common_sampler * src, common_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }

    GGML_ASSERT((src->grmr == nullptr) == (dst->grmr == nullptr));
    GGML_ASSERT((src->rbudget == nullptr) == (dst->rbudget == nullptr));

    llama_sampler_copy(src->grmr,    dst->grmr);
    llama_sampler_copy(src->rbudget, dst->rbudget);
    llama_sampler_copy(src->chain,   dst->chain);

    dst->params     = src->params;
    dst->prev       = src->prev;
    dst->cur        = src->cur;
    dst->cur_p      = src->cur_p;
    dst->cur_p.data = src->cur_p.data ? dst->cur.data() : nullptr; // re-point to dst's buffer
    dst->t_total_us = src->t_total_us;
    dst->rng_spec   = src->rng_spec;
    dst->backend_wanted = src->backend_wanted;
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    llama_synchronize(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    gsmpl->set_logits(ctx, idx);

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            // a grammar that constrains the tokens needs the full logits: callers switch backend sampling off while
            // one does (common_sampler_backend_ok); a lazy grammar before its trigger leaves the candidates as they are
            GGML_ASSERT(!grammar_constrains(gsmpl) && "using grammar in combination with backend sampling is not supported");

            // the reasoning budget forces its end sequence: p is all on the forced token
            const llama_token forced = common_reasoning_budget_get_forced(rbudget);
            if (forced != LLAMA_TOKEN_NULL) {
                gsmpl->cur.assign(1, llama_token_data{ forced, 0.0f, 1.0f });
                cur_p = { gsmpl->cur.data(), 1, 0, true };

                return forced;
            }

            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }
    }

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

namespace {
void spec_stats_record_verify(common_sampler * gsmpl, llama_context * ctx, const std::vector<int> & idxs,
        const llama_tokens & draft, const std::vector<llama_token> & result,
        const std::vector<std::vector<llama_token_data>> * draft_q = nullptr);
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }

        // a grammar starts constraining after this token (a lazy grammar's trigger), but the later positions were
        // sampled on the backend without it: stop here, this token becomes the last sampled one
        if (grammar_constrains(gsmpl) && llama_get_sampled_token_ith(ctx, idxs[i]) != LLAMA_TOKEN_NULL) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    if (common_spec_stats_enabled()) {
        spec_stats_record_verify(gsmpl, ctx, idxs, draft, result);
    }

    return result;
}

llama_token common_spec_rejection_step(const llama_token_data_array & cur_p, llama_token x,
        const std::vector<llama_token_data> & q, std::mt19937 & rng, bool & accepted) {
    accepted = false;

    auto q_of = [&q](llama_token id) -> double {
        for (const auto & e : q) {
            if (e.id == id) {
                return e.p;
            }
        }
        return 0.0;
    };

    double psum = 0.0;
    double px   = 0.0;
    for (size_t j = 0; j < cur_p.size; ++j) {
        psum += cur_p.data[j].p;
        if (cur_p.data[j].id == x) {
            px = cur_p.data[j].p;
        }
    }

    if (psum <= 0.0) {
        return LLAMA_TOKEN_NULL;
    }

    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // accept x with probability min(1, p(x)/q(x))
    const double qx = q_of(x);
    if (qx > 0.0 && unif(rng) * qx * psum < px) {
        accepted = true;
        return x;
    }

    // rejected: sample from the residual norm(max(0, p - q))
    std::vector<double> r(cur_p.size);
    double rsum = 0.0;
    for (size_t j = 0; j < cur_p.size; ++j) {
        r[j] = std::max(0.0, cur_p.data[j].p / psum - q_of(cur_p.data[j].id));
        rsum += r[j];
    }

    if (rsum <= 1e-12) {
        return LLAMA_TOKEN_NULL;
    }

    llama_token z = LLAMA_TOKEN_NULL;
    double u = unif(rng) * rsum;
    for (size_t j = 0; j < cur_p.size; ++j) {
        if (r[j] <= 0.0) {
            continue;
        }
        z = cur_p.data[j].id;
        u -= r[j];
        if (u <= 0.0) {
            break;
        }
    }

    return z;
}

bool common_sampler_supports_rejection(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    const auto & params = gsmpl->params;

    // backend sampling is fine: the backend chain exports the post-chain candidate probs, which is
    // the p the rejection step needs (the backend-sampled token is only the degenerate-p fallback).
    // A lazy grammar before its trigger leaves p as it is; the verification stops where it starts.
    return !grammar_constrains(gsmpl) &&
        params.temp > 0.0f &&
        params.mirostat == 0 &&
        params.xtc_probability <= 0.0f &&
        params.adaptive_target < 0.0f;
}

bool common_sampler_backend_ok(const struct common_sampler * gsmpl) {
    return gsmpl && gsmpl->backend_wanted && !grammar_constrains(gsmpl);
}

void common_sampler_backend_release(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_chain_backend_release(gsmpl->chain);
    gsmpl->backend_wanted = false;
}

std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(
        struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft,
        const std::vector<std::vector<llama_token_data>> & draft_q, bool is_replay) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    if (!is_replay && (draft_q.size() < draft.size() || !common_sampler_supports_rejection(gsmpl))) {
        return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, false);
    }

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    // a grammar that starts constraining within the batch (a lazy grammar's trigger) ends the verification at that
    // token: p at the later positions depends on the grammar, and backend-sampled positions have no logits for it
    bool grammar_started = false;

    for (size_t i = 0; i < draft.size() && !grammar_started; i++) {
        const llama_token x = draft[i];

        if (is_replay) {
            common_sampler_accept(gsmpl, x, true);
            result.push_back(x);
            continue;
        }

        // target distribution p at this position (the sampled token y is a fallback only)
        const llama_token y = common_sampler_sample(gsmpl, ctx, idxs[i], false);

        bool accepted = false;
        llama_token z = common_spec_rejection_step(gsmpl->cur_p, x, draft_q[i], gsmpl->rng_spec, accepted);
        if (z == LLAMA_TOKEN_NULL) {
            z = y; // degenerate p - keep the token sampled by the chain
        }

        if (accepted) {
            common_sampler_accept(gsmpl, x, true);
            result.push_back(x);
            grammar_started = grammar_constrains(gsmpl);
            continue;
        }

        common_sampler_accept(gsmpl, z, true);
        result.push_back(z);

        if (common_spec_stats_enabled()) {
            spec_stats_record_verify(gsmpl, ctx, idxs, draft, result, &draft_q);
        }

        return result;
    }

    // all drafts accepted: the bonus token is sampled from p (a replayed prefix can end on a grammar trigger; with
    // backend sampling that position has no logits to apply the grammar to, so the prefix ends the batch instead)
    const bool bonus_ok = !grammar_started &&
        !(is_replay && grammar_constrains(gsmpl) && llama_get_sampled_token_ith(ctx, idxs[draft.size()]) != LLAMA_TOKEN_NULL);

    if (bonus_ok) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[draft.size()], false);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    if (common_spec_stats_enabled()) {
        spec_stats_record_verify(gsmpl, ctx, idxs, draft, result, &draft_q);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}

//
// speculative decoding diagnostics (LLAMA_SPEC_STATS)
//

namespace {

struct spec_stats {
    static constexpr int N_POS = 16;

    bool enabled  = false;
    int  interval = 64;

    bool replay = false;

    // q of the current draft, per draft position: (token id, prob)
    std::vector<std::vector<std::pair<llama_token, float>>> q;

    int64_t n_cycles    = 0; // verify cycles with stats (non-replay)
    int64_t n_replay    = 0;
    int64_t n_tokens    = 0; // tokens emitted by the stats cycles (accepted drafts + 1)
    int64_t n_cycles_t  = 0; // all generation cycles (for timings)

    int64_t n_eval[N_POS] = {};
    double  sum_pex[N_POS] = {}; // p_target(draft token)          -> exact-match acceptance
    double  sum_ov [N_POS] = {}; // sum_x min(p_target(x), q(x))   -> rejection-sampling acceptance
    int64_t n_acc  [N_POS] = {}; // actually accepted (exact match, given the prefix was accepted)
    int64_t n_reach[N_POS] = {};

    double t_ms[COMMON_SPEC_STATS_T_COUNT] = {};

    // LLAMA_SPEC_TEMPS=0.5,0.7,1.0: re-weight the kept draft candidates (their raw logits) at each listed
    // temperature and average sum_x min(p(x), q_T(x)) on the same drafted path (rejection verification only)
    std::vector<float>  temps;
    std::vector<double> sum_ov_t; // [temps.size() * N_POS]
    int64_t n_eval_t[N_POS] = {};

    spec_stats() {
        const char * env = getenv("LLAMA_SPEC_STATS");
        if (env && atoi(env) != 0) {
            enabled = true;
            if (atoi(env) > 1) {
                interval = atoi(env);
            }
        }
        const char * env_t = getenv("LLAMA_SPEC_TEMPS");
        if (enabled && env_t) {
            for (const auto & t : string_split<std::string>(env_t, ',')) {
                temps.push_back(std::stof(t));
            }
            sum_ov_t.assign(temps.size() * N_POS, 0.0);
        }
    }

    void print() const {
        static const char * t_names[COMMON_SPEC_STATS_T_COUNT] = { "draft", "decode", "accept" };

        std::string s_pex, s_ov, s_acc;
        double len_ex = 1.0, len_rej = 1.0, prod_ex = 1.0, prod_rej = 1.0;
        for (int k = 0; k < N_POS && n_eval[k] > 0; ++k) {
            const double pex = sum_pex[k] / n_eval[k];
            const double ov  = sum_ov [k] / n_eval[k];
            const double acc = n_reach[k] > 0 ? (double) n_acc[k] / n_reach[k] : 0.0;
            prod_ex  *= pex; len_ex  += prod_ex;
            prod_rej *= ov;  len_rej += prod_rej;
            s_pex += string_format("%.3f ", pex);
            s_ov  += string_format("%.3f ", ov);
            s_acc += string_format("%.3f ", acc);
        }

        LOG_INF("spec-stats: cycles %lld (replay %lld), observed len %.3f\n",
                (long long) n_cycles, (long long) n_replay, n_cycles > 0 ? (double) n_tokens / n_cycles : 0.0);
        LOG_INF("spec-stats:   observed exact acc/pos : %s\n", s_acc.c_str());
        LOG_INF("spec-stats:   mean p(draft)  /pos    : %s-> model len %.3f (exact match)\n", s_pex.c_str(), len_ex);
        LOG_INF("spec-stats:   mean sum min(p,q)/pos  : %s-> model len %.3f (rejection sampling)\n", s_ov.c_str(), len_rej);

        for (size_t t = 0; t < temps.size(); ++t) {
            std::string s_t_ov;
            double len_t = 1.0, prod_t = 1.0;
            for (int k = 0; k < N_POS && n_eval_t[k] > 0; ++k) {
                const double ov = sum_ov_t[t * N_POS + k] / n_eval_t[k];
                prod_t *= ov; len_t += prod_t;
                s_t_ov += string_format("%.3f ", ov);
            }
            LOG_INF("spec-stats:   draft temp %.2f sum min(p,q_T)/pos : %s-> model len %.3f\n", temps[t], s_t_ov.c_str(), len_t);
        }

        std::string s_t;
        for (int k = 0; k < COMMON_SPEC_STATS_T_COUNT; ++k) {
            s_t += string_format("%s %.2f ms  ", t_names[k], n_cycles_t > 0 ? t_ms[k] / n_cycles_t : 0.0);
        }
        LOG_INF("spec-stats:   per cycle: %s\n", s_t.c_str());
    }
};

spec_stats & g_spec_stats() {
    static spec_stats s;
    return s;
}

void spec_stats_record_verify(common_sampler * gsmpl, llama_context * ctx, const std::vector<int> & idxs,
        const llama_tokens & draft, const std::vector<llama_token> & result,
        const std::vector<std::vector<llama_token_data>> * draft_q) {
    auto & st = g_spec_stats();

    if (st.replay) {
        st.n_replay++;
        return;
    }

    // evaluate the target distribution at every draft position (conditioned on the draft prefix) with a
    // clone, so the real sampler state is untouched
    common_sampler_ptr clone(common_sampler_clone(gsmpl));

    for (size_t i = 0; i < draft.size() && i < (size_t) spec_stats::N_POS; ++i) {
        common_sampler_sample(clone.get(), ctx, idxs[i], false);
        const llama_token_data_array * cur_p = common_sampler_get_candidates(clone.get(), false);

        double psum = 0.0;
        for (size_t j = 0; j < cur_p->size; ++j) {
            psum += cur_p->data[j].p;
        }
        if (psum <= 0.0) {
            continue;
        }

        // the draft's own q (rejection verification) takes precedence over the MTP stats copy
        std::vector<std::pair<llama_token, float>> q_own;
        if (draft_q && i < draft_q->size()) {
            for (const auto & e : (*draft_q)[i]) {
                q_own.emplace_back(e.id, e.p);
            }
        }

        double pex = 0.0;
        double ov  = 0.0;
        const auto * q = !q_own.empty() ? &q_own : (i < st.q.size() ? &st.q[i] : nullptr);
        for (size_t j = 0; j < cur_p->size; ++j) {
            const double p = cur_p->data[j].p / psum;
            if (cur_p->data[j].id == draft[i]) {
                pex = p;
            }
            if (q) {
                for (const auto & e : *q) {
                    if (e.first == cur_p->data[j].id) {
                        ov += std::min(p, (double) e.second);
                        break;
                    }
                }
            }
        }

        st.n_eval [i]++;
        st.sum_pex[i] += pex;
        st.sum_ov [i] += q ? ov : pex;

        if (!st.temps.empty() && draft_q && i < draft_q->size() && !(*draft_q)[i].empty()) {
            const auto & dq = (*draft_q)[i];
            float max_l = dq[0].logit;
            for (const auto & e : dq) {
                max_l = std::max(max_l, e.logit);
            }
            for (size_t t = 0; t < st.temps.size(); ++t) {
                const double inv_t = 1.0 / std::max(st.temps[t], 1e-3f);
                double sum = 0.0;
                for (const auto & e : dq) {
                    sum += exp((e.logit - max_l) * inv_t);
                }
                double ov_t = 0.0;
                for (const auto & e : dq) {
                    const double qt = exp((e.logit - max_l) * inv_t) / sum;
                    for (size_t j = 0; j < cur_p->size; ++j) {
                        if (cur_p->data[j].id == e.id) {
                            ov_t += std::min((double) cur_p->data[j].p / psum, qt);
                            break;
                        }
                    }
                }
                st.sum_ov_t[t * spec_stats::N_POS + i] += ov_t;
            }
            st.n_eval_t[i]++;
        }
    }

    // observed exact-match acceptance per position
    const size_t n_acc = result.size() - 1;
    for (size_t i = 0; i < draft.size() && i < (size_t) spec_stats::N_POS; ++i) {
        if (i > n_acc) {
            break;
        }
        st.n_reach[i]++;
        if (i < n_acc) {
            st.n_acc[i]++;
        }
    }

    st.n_cycles++;
    st.n_tokens += result.size();
}

} // namespace

bool common_spec_stats_enabled() {
    return g_spec_stats().enabled;
}

void common_spec_stats_set_draft_q(int pos, const llama_token_data_array * cur_p) {
    auto & st = g_spec_stats();
    if (!st.enabled || pos < 0 || cur_p == nullptr || cur_p->size == 0) {
        return;
    }
    if (pos == 0) {
        st.q.clear();
    }
    if ((int) st.q.size() <= pos) {
        st.q.resize(pos + 1);
    }

    // the draft sampler keeps the top-k candidates; renormalize their logits at temp 1
    std::vector<llama_token_data> c(cur_p->data, cur_p->data + cur_p->size);
    const size_t k = std::min<size_t>(c.size(), 10);
    std::partial_sort(c.begin(), c.begin() + k, c.end(), [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit;
    });

    const float max_l = c[0].logit;
    double sum = 0.0;
    for (size_t j = 0; j < k; ++j) {
        sum += expf(c[j].logit - max_l);
    }

    auto & q = st.q[pos];
    q.clear();
    for (size_t j = 0; j < k; ++j) {
        q.emplace_back(c[j].id, (float) (expf(c[j].logit - max_l) / sum));
    }
}

void common_spec_stats_set_replay(bool is_replay) {
    g_spec_stats().replay = is_replay;
}

void common_spec_stats_add_time(enum common_spec_stats_time kind, double ms) {
    auto & st = g_spec_stats();
    if (st.enabled && kind >= 0 && kind < COMMON_SPEC_STATS_T_COUNT) {
        st.t_ms[kind] += ms;
    }
}

void common_spec_stats_end_cycle() {
    auto & st = g_spec_stats();
    if (!st.enabled) {
        return;
    }
    st.n_cycles_t++;
    if (st.n_cycles_t % st.interval == 0) {
        st.print();
    }
}

void common_spec_trace(const char * label) {
    static const int interval = [] { const char * e = getenv("LLAMA_SPEC_TRACE"); return e ? atoi(e) : 0; }();
    if (interval <= 0) {
        return;
    }
    struct acc { double sum = 0; uint64_t n = 0; int order; double cpu = 0; std::vector<float> v; };
    static std::map<std::string, acc> accs;
    static const char * prev  = nullptr;
    static int64_t      t_prev = 0;
    static uint64_t     n_cyc  = 0;
    static int          n_order = 0;
    static int64_t      c_prev = 0;

    const int64_t t = ggml_time_us();
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    const int64_t c = (int64_t) ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
    if (prev) {
        const std::string key = std::string(prev) + " -> " + label;
        auto it = accs.find(key);
        if (it == accs.end()) {
            it = accs.emplace(key, acc{0, 0, n_order++}).first;
        }
        it->second.sum += 1e-3 * (t - t_prev);
        it->second.v.push_back(1e-3f * (t - t_prev));
        it->second.cpu += 1e-3 * (c - c_prev);
        it->second.n++;
    }
    prev = label;
    t_prev = t;
    c_prev = c;

    if (strcmp(label, "verify_call") == 0 && ++n_cyc % interval == 0) {
        std::vector<std::pair<int, std::string>> order;
        for (auto & kv : accs) {
            order.push_back({kv.second.order, kv.first});
        }
        std::sort(order.begin(), order.end());
        double tot = 0;
        for (auto & o : order) {
            auto & a = accs[o.second];
            std::sort(a.v.begin(), a.v.end());
            const double med = a.v.empty() ? 0.0 : a.v[a.v.size() / 2];
            LOG_INF("spec-trace: %-34s median %7.3f ms, mean %7.3f ms/cycle, thread cpu %7.3f (%.2f per cycle)\n", o.second.c_str(), med, a.sum / interval, a.cpu / interval, (double) a.n / interval);
            if ((double) a.n / interval > 0.5) { tot += med; }
        }
        LOG_INF("spec-trace: sum of medians (steps seen > 0.5 per cycle) %.3f ms\n", tot);
        accs.clear();
        n_order = 0;
    }
}
