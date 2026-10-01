# Harness G completo (replay GDN esatto), worktree fase2g + estensione gdn_replay_C.
# Catena di STEPS step; a ogni step si confronta, bit a bit, il percorso di oggi con il replay:
#  CUDA (step puri): op di oggi torch.ops._C.fused_gdn_decode_post_conv_mtp (snapshot per token)
#     F) fedelta' build: _gdn_replay.native == op di oggi (uscite + snapshot)
#     O) uscite: _gdn_replay.replay == op di oggi
#     S) stato committed (GDNRecoverSSMCommitContext.commit, sequenziale) == snapshot di oggi al token accettato
#  FLA (step misti): fused_sigmoid_gating_delta_rule_update di oggi contro gdn_recoverssm_verify (record decay) + commit
#     O) uscite, S) stato committed.
# Uso: VLLM_GDN_REPLAY_LIB=<.so> PYTHONPATH=/mnt/llmunity-models/vllm-fase2g python g_harness_full.py [steps]
import os, sys

import torch

from vllm.model_executor.layers.mamba.gdn.recoverssm_gdn import (
    GDNRecoverSSMCommitContext, gdn_recoverssm_verify)
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update

torch.ops.load_library(os.environ["VLLM_GDN_REPLAY_LIB"])
N, P, H, HV, K, V = 4, 4, 16, 48, 128, 128  # 4 richieste, finestra 1+3, teste di Flash-Next
QKV = 2 * H * K + HV * V
dev = "cuda"
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
torch.manual_seed(0)
A_log = torch.log(torch.empty(HV, device=dev).uniform_(1, 16))
dt_bias = (torch.randn(HV, device=dev) * 0.5)
norm_w = (1 + 0.1 * torch.randn(V, device=dev)).to(torch.bfloat16)
slots = 1 + N + N * P
ckpt = torch.arange(1, 1 + N, device=dev, dtype=torch.int32)                     # slot committed per richiesta
snap = torch.arange(1 + N, 1 + N + N * P, device=dev, dtype=torch.int32).view(N, P)  # snapshot di oggi
cu = torch.arange(0, N * P + 1, P, device=dev, dtype=torch.int32)
one = torch.ones(N, device=dev, dtype=torch.int32)


def commit_ctx(state, replay):
    conv = torch.zeros(slots, 8, P + 3, device=dev, dtype=torch.bfloat16)  # conv fittizio (commit_conv=False)
    return GDNRecoverSSMCommitContext.from_tensors([conv], [state], [replay], spec_query_len=P, max_num_reqs=N)


bad = {k: 0 for k in ("cuda_F", "cuda_O", "cuda_S", "fla_O", "fla_S")}
st_today = torch.zeros(slots, HV, V, K, device=dev); st_today[1:1 + N] = torch.randn(N, HV, V, K, device=dev) * 0.05
st_fla = st_today.clone()
for step in range(steps):
    acc = torch.randint(1, P + 1, (N,), device=dev, dtype=torch.int32)
    qkv = torch.randn(N * P, QKV, device=dev, dtype=torch.bfloat16)
    a = torch.randn(N * P, HV, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N * P, HV, device=dev, dtype=torch.bfloat16)
    gate = torch.randn(N * P, HV, V, device=dev, dtype=torch.bfloat16)
    pick = snap.gather(1, (acc - 1).long()[:, None]).squeeze(1).long()

    # --- CUDA, step puri ---
    idx = snap.clone()
    base = st_today.clone(); base[snap[:, 0].long()] = st_today[ckpt.long()]
    s_now, s_nat = base.clone(), base.clone()
    o_now = torch.empty(N * P, HV, V, device=dev, dtype=torch.bfloat16); o_nat = torch.empty_like(o_now)
    torch.ops._C.fused_gdn_decode_post_conv_mtp(qkv, a, b, A_log, dt_bias, idx, cu, one, s_now, gate, norm_w, o_now, K ** -0.5, 1e-5, "silu")
    torch.ops._gdn_replay.native(qkv, a, b, A_log, dt_bias, idx, cu, one, s_nat, gate, norm_w, o_nat, K ** -0.5, 1e-5, "silu")
    bad["cuda_F"] += not (torch.equal(o_now, o_nat) and torch.equal(s_now, s_nat))
    s_rep = st_today.clone(); rec = torch.zeros(slots, HV, P, V + K + 1, device=dev)
    o_rep = torch.empty_like(o_now)
    torch.ops._gdn_replay.replay(qkv, a, b, A_log, dt_bias, ckpt[:, None].contiguous(), cu, s_rep, rec, gate, norm_w, o_rep, K ** -0.5, 1e-5, "silu")
    bad["cuda_O"] += not torch.equal(o_rep, o_now)
    commit_ctx(s_rep, rec).commit(acc, ckpt, cu, commit_conv=False)
    today = s_now[pick]
    ok = torch.equal(s_rep[ckpt.long()], today)
    if not ok and bad["cuda_S"] < 3:
        d = (s_rep[ckpt.long()] - today).abs()
        print(f"cuda step {step}: {(d > 0).sum().item()} diversi, max {d.max().item():.3e}, acc {acc.tolist()}")
    bad["cuda_S"] += not ok
    st_today[ckpt.long()] = today

    # --- FLA, step misti ---
    q = qkv[:, :H * K].view(1, -1, H, K).contiguous(); k = qkv[:, H * K:2 * H * K].view(1, -1, H, K).contiguous()
    v = qkv[:, 2 * H * K:].view(1, -1, HV, V).contiguous()
    base = st_fla.clone(); base[snap[:, 0].long()] = st_fla[ckpt.long()]
    o_fla, _ = fused_sigmoid_gating_delta_rule_update(A_log, a.view(1, -1, HV), b.view(1, -1, HV), dt_bias, q, k, v,
                                                      initial_state=base, inplace_final_state=True, cu_seqlens=cu,
                                                      ssm_state_indices=snap, num_accepted_tokens=one,
                                                      use_qk_l2norm_in_kernel=True)
    s_rep = st_fla.clone(); rec = torch.zeros(slots, HV, P, V + K + 1, device=dev)
    o_v = gdn_recoverssm_verify(A_log, a, b, dt_bias, q, k, v, checkpoint_state=s_rep, replay_cache=rec,
                                query_start_loc=cu, state_indices=ckpt, spec_query_len=P, use_qk_l2norm_in_kernel=True)
    bad["fla_O"] += not torch.equal(o_v.view_as(o_fla), o_fla)
    commit_ctx(s_rep, rec).commit(acc, ckpt, cu, commit_conv=False)
    today = base[pick]
    ok = torch.equal(s_rep[ckpt.long()], today)
    if not ok and bad["fla_S"] < 3:
        d = (s_rep[ckpt.long()] - today).abs()
        print(f"fla step {step}: {(d > 0).sum().item()} diversi, max {d.max().item():.3e}, acc {acc.tolist()}")
    bad["fla_S"] += not ok
    st_fla[ckpt.long()] = today

print(" | ".join(f"{k}: {steps - v}/{steps}" for k, v in bad.items()))
print("GATE G (harness):", "PASSATO" if not any(bad.values()) else "FALLITO")
sys.exit(1 if any(bad.values()) else 0)
