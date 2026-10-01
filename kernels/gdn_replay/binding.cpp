// Estensione separata per l'harness G: stesso .cu del fork + replay (#59366, record con decay).
// Op: _gdn_replay::native (= fused_gdn_decode_post_conv_mtp di oggi) e _gdn_replay::replay.
#include <string>
#include <torch/csrc/stable/library.h>
#include "libtorch_stable/torch_utils.h"
#include "core/registration.h"

void fused_gdn_decode_post_conv_mtp(
    torch::stable::Tensor const& mixed_qkv, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_log,
    torch::stable::Tensor const& dt_bias,
    torch::stable::Tensor const& state_indices,
    torch::stable::Tensor const& cu_seqlens,
    torch::stable::Tensor const& num_accepted_tokens,
    torch::stable::Tensor& state, torch::stable::Tensor const& output_gate,
    torch::stable::Tensor const& norm_weight, torch::stable::Tensor& out,
    double scale, double norm_eps, const std::string& output_gate_activation);

void fused_gdn_decode_post_conv_mtp_replay(
    torch::stable::Tensor const& mixed_qkv, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_log,
    torch::stable::Tensor const& dt_bias,
    torch::stable::Tensor const& state_indices,
    torch::stable::Tensor const& cu_seqlens, torch::stable::Tensor& state,
    torch::stable::Tensor& replay, torch::stable::Tensor const& output_gate,
    torch::stable::Tensor const& norm_weight, torch::stable::Tensor& out,
    double scale, double norm_eps, const std::string& output_gate_activation);

STABLE_TORCH_LIBRARY_FRAGMENT(_gdn_replay, ops) {
  ops.def(
      "native(Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
      "Tensor state_indices, Tensor cu_seqlens, Tensor num_accepted_tokens, "
      "Tensor! state, Tensor output_gate, Tensor norm_weight, Tensor! out, "
      "float scale, float norm_eps=1e-5, str output_gate_activation='silu') -> ()");
  ops.def(
      "replay(Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
      "Tensor state_indices, Tensor cu_seqlens, Tensor state, Tensor! replay, "
      "Tensor output_gate, Tensor norm_weight, Tensor! out, "
      "float scale, float norm_eps=1e-5, str output_gate_activation='silu') -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_gdn_replay, CUDA, ops) {
  ops.impl("native", TORCH_BOX(&fused_gdn_decode_post_conv_mtp));
  ops.impl("replay", TORCH_BOX(&fused_gdn_decode_post_conv_mtp_replay));
}
