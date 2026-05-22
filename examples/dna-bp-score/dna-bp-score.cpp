// Base-pair scoring for Carbon/HybridDNA models: for a DNA sequence wrapped in
// <dna>, report P(base | preceding context) at each position, marginalized from
// the k-mer distribution exactly like the sampler. Mirrors the Python
// HybridDNATokenizer model's score_sequence().

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <array>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int dna_base_index(char c) {
    switch (c) {
        case 'A': return 0;
        case 'T': return 1;
        case 'C': return 2;
        case 'G': return 3;
        default:  return -1;
    }
}

int main(int argc, char ** argv) {
    common_params params;
    params.prompt = "GGGCTATAAAGGCCATCGATCGATCGATCGATCGATCGATCG";
    params.n_ctx  = 0;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // the prompt is the raw DNA sequence (ACGT), without the <dna> tag
    std::string seq;
    for (char c : params.prompt) {
        seq += (c >= 'a' && c <= 'z') ? char(c - 32) : c;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto            llama_init = common_init_from_params(params);
    llama_model   * model      = llama_init->model();
    llama_context * ctx        = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // locate the k-mer block: tokenize "<dna>AAAA...A" to recover the first
    // k-mer id and k (its text length), then n_kmers = 4^k
    const llama_token first = [&] {
        const auto t = common_tokenize(vocab, "<dna>AAAAAAAAAAAA", false, true);
        return t.size() >= 2 ? t[1] : LLAMA_TOKEN_NULL;
    }();
    if (first == LLAMA_TOKEN_NULL) {
        LOG_ERR("%s: could not locate DNA k-mer block (not a HybridDNA vocab?)\n", __func__);
        return 1;
    }
    const int32_t k = (int32_t) std::strlen(llama_vocab_get_text(vocab, first));
    std::vector<int32_t> pow4(k);
    int32_t n_kmers = 1;
    for (int32_t i = k - 1; i >= 0; --i) {
        pow4[i] = n_kmers;
        n_kmers *= 4;
    }

    // right-pad the sequence to a multiple of k with 'A' (training convention)
    std::string padded = seq;
    if (padded.size() % k != 0) {
        padded.append(k - padded.size() % k, 'A');
    }
    const int32_t n_tok = (int32_t) padded.size() / k;

    // tokenize "<dna>" + padded sequence
    const std::vector<llama_token> tokens = common_tokenize(vocab, "<dna>" + padded, false, true);

    // decode, requesting logits at every position that predicts a k-mer
    // (position t predicts the token at t+1; position 0 is <dna> -> k-mer 0)
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (int32_t i = 0; i < (int32_t) tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], i, {0}, i < n_tok);
    }
    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    printf("# pos base P(base|context)\n");
    for (int32_t t = 0; t < n_tok; ++t) {
        const float * logits = llama_get_logits_ith(ctx, t);

        // softmax over the k-mer block (shifted by its running max)
        float maxl = -INFINITY;
        for (int32_t f = 0; f < n_kmers && first + f < n_vocab; ++f) {
            maxl = std::max(maxl, logits[first + f]);
        }
        std::vector<std::array<float, 4>> bp(k, {0.0f, 0.0f, 0.0f, 0.0f});
        float sum = 0.0f;
        for (int32_t f = 0; f < n_kmers && first + f < n_vocab; ++f) {
            const float w = expf(logits[first + f] - maxl);
            sum += w;
            for (int32_t pos = 0; pos < k; ++pos) {
                bp[pos][(f / pow4[pos]) % 4] += w;
            }
        }

        for (int32_t pos = 0; pos < k; ++pos) {
            const int32_t gi = t * k + pos;
            if (gi >= (int32_t) seq.size()) {
                break;
            }
            const int b = dna_base_index(seq[gi]);
            const float p = (b < 0 || sum == 0.0f) ? 0.0f : bp[pos][b] / sum;
            printf("%4d %c %.4f\n", gi, seq[gi], p);
        }
    }

    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
