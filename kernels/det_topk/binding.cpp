// Estensione separata: persistent_topk con la selezione deterministica di vLLM #55122 (backport, solo se serve).
// Op: _det_topk::persistent_topk, stessa firma dell'op _C di oggi.
#include <torch/csrc/stable/library.h>
#include "libtorch_stable/torch_utils.h"
#include "core/registration.h"

void persistent_topk(const torch::stable::Tensor& logits, const torch::stable::Tensor& lengths,
                     torch::stable::Tensor& output, torch::stable::Tensor& workspace, int64_t k,
                     int64_t max_seq_len);

STABLE_TORCH_LIBRARY_FRAGMENT(_det_topk, ops) {
  ops.def("persistent_topk(Tensor logits, Tensor lengths, Tensor! output, Tensor workspace, int k, int max_seq_len) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_det_topk, CUDA, ops) {
  ops.impl("persistent_topk", TORCH_BOX(&persistent_topk));
}
