// lossless speculative sampling: x ~ q, then common_spec_rejection_step must emit tokens distributed exactly as p,
// with acceptance rate sum_x min(p(x), q(x))

#include "sampling.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <random>
#include <vector>

static int test_case(const char * name, std::vector<llama_token_data> p, const std::vector<llama_token_data> & q, int n) {
    std::mt19937 rng_q(1234);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    double psum = 0.0;
    for (const auto & e : p) {
        psum += e.p;
    }

    double expected_acc = 0.0;
    for (const auto & e : p) {
        for (const auto & f : q) {
            if (f.id == e.id) {
                expected_acc += std::min((double) e.p / psum, (double) f.p);
            }
        }
    }

    std::map<llama_token, int> hist;
    int n_acc = 0;

    for (int it = 0; it < n; ++it) {
        // x ~ q
        llama_token x = q.back().id;
        double u = unif(rng_q);
        for (const auto & f : q) {
            u -= f.p;
            if (u <= 0.0) {
                x = f.id;
                break;
            }
        }

        llama_token_data_array cur_p = { p.data(), p.size(), -1, false };
        bool accepted = false;
        const llama_token z = common_spec_rejection_step(cur_p, x, q, rng, accepted);
        if (z == LLAMA_TOKEN_NULL) {
            fprintf(stderr, "%s: unexpected degenerate result\n", name);
            return 1;
        }
        n_acc += accepted ? 1 : 0;
        hist[z]++;
    }

    int ret = 0;
    const double tol = 5.0 / std::sqrt((double) n); // ~5 sigma at p = 0.25

    for (const auto & e : p) {
        const double want = e.p / psum;
        const double got  = (double) hist[e.id] / n;
        if (std::fabs(want - got) > tol) {
            fprintf(stderr, "%s: token %d: p = %.4f, emitted %.4f\n", name, e.id, want, got);
            ret = 1;
        }
        hist.erase(e.id);
    }
    for (const auto & kv : hist) {
        if (kv.second > 0) {
            fprintf(stderr, "%s: emitted token %d outside the support of p (%d times)\n", name, kv.first, kv.second);
            ret = 1;
        }
    }

    const double acc = (double) n_acc / n;
    if (std::fabs(acc - expected_acc) > tol) {
        fprintf(stderr, "%s: acceptance %.4f, expected %.4f\n", name, acc, expected_acc);
        ret = 1;
    }

    printf("%s: acceptance %.4f (expected %.4f) %s\n", name, acc, expected_acc, ret == 0 ? "OK" : "FAIL");
    return ret;
}

// block verification (common_spec_block_plan / common_spec_block_residual): the exact distribution of the first N
// emitted tokens, by enumerating every draft path, accepted length and next token over several cycles on a toy model
// whose target and draft distributions are random per prefix, must equal the target's autoregressive distribution.
// A per-position check cannot catch a wrong residual or threshold here: the test is over the joint distribution.
struct toy_model {
    int      n_vocab;
    uint64_t seed;
    double   zero_frac; // share of tokens with probability 0 (as left by top-k / top-p truncation)
    bool     copy_of_other = false;
    toy_model * other = nullptr;
    std::map<std::vector<int>, std::vector<double>> cache;

    toy_model(int nv, uint64_t sd, double zf) : n_vocab(nv), seed(sd), zero_frac(zf) {}

    const std::vector<double> & dist(const std::vector<int> & prefix) {
        if (copy_of_other) {
            return other->dist(prefix);
        }
        auto it = cache.find(prefix);
        if (it != cache.end()) {
            return it->second;
        }
        uint64_t h = seed;
        for (int t : prefix) {
            h = h * 1000003ull + (uint64_t) t + 1;
        }
        std::mt19937_64 g(h);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<double> d(n_vocab);
        double s = 0.0;
        for (int v = 0; v < n_vocab; ++v) {
            const double z = u(g);
            const double x = u(g);
            d[v] = z < zero_frac ? 0.0 : x * x; // squared: some peaked distributions
            s += d[v];
        }
        if (s <= 0.0) {
            d[g() % n_vocab] = 1.0;
            s = 1.0;
        }
        for (auto & x : d) {
            // keep exactly what a float llama_token_data can hold, so the check compares like with like
            x = (double) (float) (x / s);
        }
        return cache.emplace(prefix, d).first->second;
    }
};

static std::vector<llama_token_data> toy_sparse(const std::vector<double> & d) {
    std::vector<llama_token_data> r;
    for (size_t v = 0; v < d.size(); ++v) {
        if (d[v] > 0.0) {
            r.push_back({ (llama_token) v, 0.0f, (float) d[v] });
        }
    }
    return r;
}

struct bv_stats {
    double e_tokens_bv = 0.0; // expected tokens emitted by the first cycle
    double e_tokens_tl = 0.0; // same for token-by-token verification on the same draft paths
    double max_lost    = 0.0; // largest probability of an accepted length whose residual is empty
};

static void bv_enumerate(toy_model & tgt, toy_model & dft, int n_draft, int N, const std::vector<int> & out, double w,
        std::map<std::vector<int>, double> & res, bv_stats & st, bool first) {
    if ((int) out.size() >= N) {
        res[std::vector<int>(out.begin(), out.begin() + N)] += w;
        return;
    }
    const int V = tgt.n_vocab;
    std::vector<int> x(n_draft);
    std::function<void(int, double)> rec = [&](int k, double wq) {
        if (k < n_draft) {
            std::vector<int> pre = out;
            pre.insert(pre.end(), x.begin(), x.begin() + k);
            const auto & qd = dft.dist(pre);
            for (int v = 0; v < V; ++v) {
                if (qd[v] > 0.0) {
                    x[k] = v;
                    rec(k + 1, wq * qd[v]);
                }
            }
            return;
        }
        std::vector<std::vector<llama_token_data>> P(n_draft + 1), Q(n_draft);
        std::vector<int> pre = out;
        for (int j = 0; j <= n_draft; ++j) {
            P[j] = toy_sparse(tgt.dist(pre));
            if (j < n_draft) {
                Q[j] = toy_sparse(dft.dist(pre));
                pre.push_back(x[j]);
            }
        }
        const llama_tokens xt(x.begin(), x.end());
        std::vector<double> h, wp;
        common_spec_block_plan(P, Q, xt, n_draft, h, wp);

        if (first) {
            double tl = 1.0, prod = 1.0;
            for (int j = 0; j < n_draft; ++j) {
                double px = 0.0, qx = 0.0;
                for (const auto & e : P[j]) if (e.id == x[j]) px = e.p;
                for (const auto & e : Q[j]) if (e.id == x[j]) qx = e.p;
                prod *= std::min(1.0, px / qx);
                tl += prod;
            }
            st.e_tokens_tl += wq * tl;
        }

        for (int tau = 0; tau <= n_draft; ++tau) {
            // P(accepted length = tau) = h[tau] prod_{j > tau} (1 - h[j]), with h[0] = 1
            double pt = tau == 0 ? 1.0 : h[tau];
            for (int j = tau + 1; j <= n_draft; ++j) {
                pt *= 1.0 - h[j];
            }
            if (pt <= 0.0) {
                continue;
            }
            if (first) {
                st.e_tokens_bv += wq * pt * (tau + 1);
            }
            std::vector<llama_token_data> r;
            common_spec_block_residual(P, Q, wp, n_draft, tau, r);
            double rs = 0.0;
            for (const auto & e : r) rs += e.p;
            if (rs <= 1e-12) {
                st.max_lost = std::max(st.max_lost, pt);
                continue;
            }
            for (const auto & e : r) {
                if (e.p <= 0.0f) {
                    continue;
                }
                std::vector<int> o2 = out;
                o2.insert(o2.end(), x.begin(), x.begin() + tau);
                o2.push_back(e.id);
                bv_enumerate(tgt, dft, n_draft, N, o2, wq * pt * e.p / rs, res, st, false);
            }
        }
    };
    rec(0, w);
}

static int test_block(const char * name, int V, int n_draft, int N, uint64_t seed, double zf_tgt, double zf_dft, bool same = false) {
    toy_model tgt(V, seed, zf_tgt);
    toy_model dft(V, seed * 7919 + 17, zf_dft);
    if (same) {
        dft.copy_of_other = true;
        dft.other = &tgt;
    }

    std::map<std::vector<int>, double> res;
    bv_stats st;
    bv_enumerate(tgt, dft, n_draft, N, {}, 1.0, res, st, true);

    // the target's own distribution of the first N tokens
    double max_err = 0.0, total = 0.0;
    std::vector<int> s(N, 0);
    std::function<void(int, double)> walk = [&](int t, double pr) {
        if (t == N) {
            const auto it = res.find(s);
            const double got = it == res.end() ? 0.0 : it->second;
            max_err = std::max(max_err, std::fabs(got - pr));
            return;
        }
        std::vector<int> pre(s.begin(), s.begin() + t);
        const auto & d = tgt.dist(pre);
        for (int v = 0; v < V; ++v) {
            s[t] = v;
            walk(t + 1, pr * d[v]);
        }
    };
    walk(0, 1.0);
    for (const auto & kv : res) {
        total += kv.second;
    }

    // float probabilities sum to 1 only within ~1e-7 per distribution
    const bool ok = max_err < 2e-6 && std::fabs(total - 1.0) < 2e-6 && st.max_lost < 1e-9 && st.e_tokens_bv >= st.e_tokens_tl - 1e-9;
    printf("%s: V %d, %d drafts, first %d tokens: max |error| %.2e, total %.9f, tokens per cycle %.4f (token test %.4f) %s\n",
            name, V, n_draft, N, max_err, total, st.e_tokens_bv, st.e_tokens_tl, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    const int n = 2000000;
    int ret = 0;

    ret |= test_block("block dense",       3, 2, 3, 11, 0.0, 0.0);
    ret |= test_block("block truncated",   3, 2, 3, 12, 0.3, 0.2);
    ret |= test_block("block 3 drafts",    3, 3, 3, 13, 0.25, 0.25);
    ret |= test_block("block 3 drafts b",  3, 3, 3, 21, 0.0, 0.4);
    ret |= test_block("block 4 tokens",    4, 2, 3, 14, 0.25, 0.1);
    ret |= test_block("block 1 draft",     4, 1, 3, 15, 0.2, 0.2);
    ret |= test_block("block same model",  3, 3, 3, 16, 0.2, 0.0, true);

    // q puts mass on a token outside the support of p (id 5), p is unsorted
    ret |= test_case("disjoint",
            { {2, 0.0f, 0.15f}, {0, 0.0f, 0.40f}, {4, 0.0f, 0.05f}, {1, 0.0f, 0.30f}, {3, 0.0f, 0.10f} },
            { {0, 0.0f, 0.10f}, {1, 0.0f, 0.50f}, {2, 0.0f, 0.30f}, {5, 0.0f, 0.10f} }, n);

    // q == p: every draft is accepted
    ret |= test_case("equal",
            { {7, 0.0f, 0.6f}, {8, 0.0f, 0.3f}, {9, 0.0f, 0.1f} },
            { {7, 0.0f, 0.6f}, {8, 0.0f, 0.3f}, {9, 0.0f, 0.1f} }, n);

    // unnormalized p (as left by a truncating sampler chain), peaked q
    ret |= test_case("unnormalized",
            { {0, 0.0f, 0.1f}, {1, 0.0f, 0.1f}, {2, 0.0f, 0.2f}, {3, 0.0f, 0.1f} },
            { {2, 0.0f, 0.9f}, {3, 0.0f, 0.05f}, {6, 0.0f, 0.05f} }, n);

    return ret;
}
