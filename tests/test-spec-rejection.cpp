// lossless speculative sampling: x ~ q, then common_spec_rejection_step must emit tokens distributed exactly as p,
// with acceptance rate sum_x min(p(x), q(x))

#include "sampling.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

int main() {
    const int n = 2000000;
    int ret = 0;

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
