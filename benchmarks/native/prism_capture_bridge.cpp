// Narrow C ABI for pinned Prism libllama capture. No graph/weight modification.
// Compile against PrismML-Eng/llama.cpp
// e3de715222b2667c54a267441159a5729f11155e.
#include "ggml-backend.h"
#include "llama-ext.h"
#include "llama.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error;
std::once_flag initialized;
struct Capture {
  llama_model *model = nullptr;
  llama_context *ctx = nullptr;
  int width = 0, vocab = 0, n_layers = 0, taps = 0, chunk = 0, context = 0;
  ~Capture() {
    if (ctx)
      llama_free(ctx);
    if (model)
      llama_model_free(model);
  }
};
} // namespace
extern "C" {
const char *draftfit_prism_revision() {
  return "e3de715222b2667c54a267441159a5729f11155e";
}
const char *draftfit_prism_error() { return error.c_str(); }
void *draftfit_prism_open(const char *path, const int32_t *layers, int count,
                          int context, int chunk, int gpu) {
  try {
    error.clear();
    if (!path || !layers || count < 1 || count > 64 || context < 64 ||
        context > 8192 || chunk < 1 || chunk > 512 || (gpu != 0 && gpu != 1))
      throw std::runtime_error("invalid native capture dimensions");
    std::call_once(initialized, []() {
      llama_backend_init();
      ggml_backend_load_all();
    });
    auto handle = std::make_unique<Capture>();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = gpu ? -1 : 0;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.main_gpu = gpu ? 0 : -1;
    handle->model = llama_model_load_from_file(path, mp);
    if (!handle->model)
      throw std::runtime_error("GGUF model load failed");
    char arch[64]{};
    const auto arch_size = llama_model_meta_val_str(
        handle->model, "general.architecture", arch, sizeof(arch));
    if (arch_size < 0 || arch_size >= int(sizeof(arch)) ||
        std::string(arch) != "qwen35")
      throw std::runtime_error("native capture supports qwen35 only");
    handle->width = llama_model_n_embd(handle->model);
    handle->vocab = llama_vocab_n_tokens(llama_model_get_vocab(handle->model));
    handle->n_layers = llama_model_n_layer(handle->model);
    if (handle->width != 5120 || handle->vocab != 248320 ||
        handle->n_layers != 64)
      throw std::runtime_error(
          "native target geometry differs from Bonsai contract");
    for (int i = 0; i < count; ++i)
      if (layers[i] < 0 || layers[i] >= handle->n_layers ||
          (i && layers[i] <= layers[i - 1]))
        throw std::runtime_error(
            "capture taps must be unique ascending block IDs");
    auto cp = llama_context_default_params();
    if (mp.n_gpu_layers == 0) {
      cp.offload_kqv = false;
      cp.op_offload = false;
    }
    // Prism's recurrent Metal graph is unstable with a reserve smaller
    // than 64 rows (observed repeat/reset divergence at n_batch=32).
    // Decodes can still consume smaller chunks; reserve the validated floor.
    const int reserve = std::max(64, chunk);
    cp.n_ctx = context;
    cp.n_batch = reserve;
    cp.n_ubatch = reserve;
    cp.n_seq_max = 1;
    cp.n_outputs_max = reserve;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.embeddings = true;
    handle->ctx = llama_init_from_model(handle->model, cp);
    if (!handle->ctx)
      throw std::runtime_error("GGUF context creation failed");
    llama_set_capture_layers(handle->ctx, layers, count);
    // Keep the validated graph output contract: normalized final state and
    // tapped block inputs remain materialized across recurrent graph reuse.
    llama_set_embeddings_nextn(handle->ctx, true, true);
    llama_set_embeddings_layer_inp(handle->ctx, 0, true);
    for (int i = 0; i < count; ++i)
      llama_set_embeddings_layer_inp(handle->ctx, layers[i] + 1, true);
    handle->taps = count;
    handle->chunk = chunk;
    handle->context = context;
    return handle.release();
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
}
void draftfit_prism_close(void *ptr) { delete static_cast<Capture *>(ptr); }
int draftfit_prism_geometry(void *ptr, int *out) {
  if (!ptr || !out)
    return -1;
  auto *h = static_cast<Capture *>(ptr);
  out[0] = h->width;
  out[1] = h->vocab;
  out[2] = h->n_layers;
  return 0;
}
int draftfit_prism_capture(void *ptr, const int32_t *ids, int n, float *aux,
                           float *hidden, float *logits) {
  if (!ptr) {
    error = "closed capture context";
    return -1;
  }
  auto *h = static_cast<Capture *>(ptr);
  llama_batch batch{};
  bool allocated = false;
  try {
    error.clear();
    if (!ids || !aux || !hidden || n < 1 || n > h->context)
      throw std::runtime_error("invalid capture buffers/length");
    for (int i = 0; i < n; ++i)
      if (ids[i] < 0 || ids[i] >= h->vocab)
        throw std::runtime_error("token outside vocabulary");
    // Full clear includes both recurrent and attention state between samples.
    llama_memory_clear(llama_get_memory(h->ctx), true);
    batch = llama_batch_init(h->chunk, 0, 1);
    allocated = true;
    for (int start = 0; start < n; start += h->chunk) {
      batch.n_tokens = std::min(h->chunk, n - start);
      for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = ids[start + i];
        batch.pos[i] = start + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
      }
      if (llama_decode(h->ctx, batch) != 0)
        throw std::runtime_error("GGUF decode failed");
      for (int i = 0; i < batch.n_tokens; ++i) {
        auto *a = llama_get_embeddings_capture_ith(h->ctx, i);
        auto *z = llama_get_embeddings_ith(h->ctx, i);
        if (!a || !z)
          throw std::runtime_error("missing native capture output");
        std::memcpy(aux + size_t(start + i) * h->width * h->taps, a,
                    size_t(h->width) * h->taps * sizeof(float));
        std::memcpy(hidden + size_t(start + i) * h->width, z,
                    size_t(h->width) * sizeof(float));
        if (logits) {
          auto *l = llama_get_logits_ith(h->ctx, i);
          if (!l)
            throw std::runtime_error("missing native logits");
          std::memcpy(logits + size_t(start + i) * h->vocab, l,
                      size_t(h->vocab) * sizeof(float));
        }
      }
    }
    llama_batch_free(batch);
    return 0;
  } catch (const std::exception &e) {
    if (allocated)
      llama_batch_free(batch);
    error = e.what();
    return -1;
  }
}
}
