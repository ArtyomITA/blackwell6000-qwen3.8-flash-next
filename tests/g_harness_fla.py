# Harness G (replay GDN esatto), parte Triton FLA (step misti), nessuna build.
# Domanda: lo stato che il kernel FLA di oggi scrive dopo il token a-1 (snapshot per token) e' bit a bit uguale a quello
# ricostruito da un commit SEQUENZIALE che riapplica i record (decay, k normalizzata, delta) scritti dallo stesso kernel?
#   oggi:   h = h * exp(g); v' = (v - sum(h*k)) * beta; h += v'[:,None] * k[None,:]   (stessa riga, Triton contrae in fma)
#   commit: h = h * decay_t; h = fma(delta_t, k_t, h)
# Catena di STEPS step: N richieste x finestra T=1+num_spec, accettati casuali 1..T, si riparte dallo stato committed.
# Uso: python g_harness_fla.py [steps]   (GPU, sotto flock). Exit 0 = bit a bit su tutti gli step.
import sys

import torch

from vllm.triton_utils import tl, triton
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update

N, T, H, HV, K, V = 4, 4, 16, 48, 128, 128
R = V + K + 1  # record: delta (V), k normalizzata (K), decay (1)


@triton.jit
def record_kernel(A_log, a, b, dt_bias, k, v, h0, idx0, rec, snapout, beta, threshold,
                  T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                  BV: tl.constexpr, R: tl.constexpr):
    # Copia del corpo di fused_sigmoid_gating_delta_rule_update_kernel (ramo spec, L2 norm, non KDA), con le stesse
    # espressioni; in piu' scrive il record per token. Le uscite o e gli snapshot non servono qui.
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    s = tl.load(idx0 + i_n).to(tl.int64)
    b_h = tl.zeros([BV, K], dtype=tl.float32)
    b_h += tl.load(h0 + s * HV * V * K + i_hv * V * K + o_v[:, None] * K + o_k[None, :]).to(tl.float32)
    for i_t in range(0, T):
        tok = i_n * T + i_t
        b_k = tl.load(k + (tok * H + i_h) * K + o_k).to(tl.float32)
        b_v = tl.load(v + (tok * HV + i_hv) * V + o_v).to(tl.float32)
        b_b = tl.load(b + tok * HV + i_hv).to(tl.float32)
        x = tl.load(a + tok * HV + i_hv).to(tl.float32) + tl.load(dt_bias + i_hv).to(tl.float32)
        softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
        b_g = -tl.exp(tl.load(A_log + i_hv).to(tl.float32)) * softplus_x
        b_beta = tl.sigmoid(b_b.to(tl.float32))
        b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        decay = tl.exp(b_g)
        b_h *= decay
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        p = rec + ((i_n * HV + i_hv) * T + i_t) * R
        tl.store(p + o_v, b_v)
        tl.store(p + V + o_k, b_k)
        tl.store(p + V + K, decay)
        tl.store(snapout + (((i_n * HV + i_hv) * T + i_t) * V + o_v[:, None]) * K + o_k[None, :], b_h)


@triton.jit
def commit_kernel(h0, idx0, dst, rec, n_acc, T: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                  BV: tl.constexpr, R: tl.constexpr, FMA_DK: tl.constexpr):
    # Commit sequenziale: riapplica i token accettati con lo stesso ordine di arrotondamento del kernel di oggi.
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    s = tl.load(idx0 + i_n).to(tl.int64)
    d = tl.load(dst + i_n).to(tl.int64)
    sp = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    h = tl.load(h0 + s * HV * V * K + sp).to(tl.float32)
    n = tl.load(n_acc + i_n)
    for t in range(0, n):
        p = rec + ((i_n * HV + i_hv) * T + t) * R
        c = tl.load(p + o_v)
        kk = tl.load(p + V + o_k)
        e = tl.load(p + V + K)
        if FMA_DK:
            h = tl.fma(c[:, None], kk[None, :], h * e)
        else:  # variante: fma(h, decay, delta*k)
            h = tl.fma(h, e, c[:, None] * kk[None, :])
    tl.store(h0 + d * HV * V * K + sp, h)


def main(steps, FMA_DK):
    torch.manual_seed(0)
    dev = "cuda"
    A_log = torch.log(torch.empty(HV, device=dev).uniform_(1, 16))
    dt_bias = torch.randn(HV, device=dev) * 0.5
    slots = 1 + N * (T + 2)
    state = torch.zeros(slots, HV, V, K, device=dev)
    state[1:1 + N] = torch.randn(N, HV, V, K, device=dev) * 0.05
    ref = state.clone()
    bad = 0
    copy_bad = 0
    for step in range(steps):
        q = torch.randn(N * T, H, K, device=dev, dtype=torch.bfloat16)
        k = torch.randn(N * T, H, K, device=dev, dtype=torch.bfloat16)
        v = torch.randn(N * T, HV, V, device=dev, dtype=torch.bfloat16)
        a = torch.randn(N * T, HV, device=dev, dtype=torch.bfloat16)
        b = torch.randn(N * T, HV, device=dev, dtype=torch.bfloat16)
        acc = torch.randint(1, T + 1, (N,), device=dev, dtype=torch.int32)
        # oggi: snapshot per token negli slot 1+N+i*T+t, partendo dallo slot committed 1+i (num_accepted_tokens=1)
        snap = torch.arange(1 + N, 1 + N + N * T, device=dev, dtype=torch.int32).view(N, T)
        idx = torch.cat([torch.arange(1, 1 + N, device=dev, dtype=torch.int32)[:, None], snap[:, 1:]], 1)
        idx_today = idx.clone(); idx_today[:, 0] = snap[:, 0]
        h0_today = ref.clone(); h0_today[snap[:, 0].long()] = ref[1:1 + N]
        cu = torch.arange(0, N * T + 1, T, device=dev, dtype=torch.int32)
        fused_sigmoid_gating_delta_rule_update(A_log, a.view(1, -1, HV), b.view(1, -1, HV), dt_bias,
                                               q.view(1, -1, H, K), k.view(1, -1, H, K), v.view(1, -1, HV, V),
                                               initial_state=h0_today, inplace_final_state=True, cu_seqlens=cu,
                                               ssm_state_indices=idx_today, num_accepted_tokens=torch.ones(N, device=dev, dtype=torch.int32),
                                               use_qk_l2norm_in_kernel=True)
        today = h0_today[snap.gather(1, (acc - 1).long()[:, None]).squeeze(1).long()]
        # replay: record + commit sequenziale nello slot committed
        rec = torch.empty(N, HV, T, R, device=dev)
        snapout = torch.empty(N, HV, T, V, K, device=dev)
        idx0 = torch.arange(1, 1 + N, device=dev, dtype=torch.int32)
        record_kernel[(V // 32, N * HV)](A_log, a, b, dt_bias, k, v, state, idx0, rec, snapout, 1.0, 20.0,
                                         T=T, H=H, HV=HV, K=K, V=V, BV=32, R=R, num_warps=4, num_stages=3)
        commit_kernel[(V // 32, N * HV)](state, idx0, idx0, rec, acc, T=T, HV=HV, K=K, V=V, BV=32, R=R,
                                         FMA_DK=FMA_DK, num_warps=4)
        mine = state[1:1 + N]
        own = snapout[torch.arange(N, device=dev), :, (acc - 1).long()]
        if not torch.equal(own, today):
            copy_bad += 1
        if not torch.equal(today, mine):
            bad += 1
            if bad <= 3:
                dif = (today != mine).sum().item()
                print(f"step {step}: {dif} elementi diversi, max |d| {(today - mine).abs().max().item():.3e}, acc {acc.tolist()}")
        ref[1:1 + N] = today
        state[1:1 + N] = today  # si riparte dallo stato di oggi: ogni step e' un test indipendente
    print(f"FLA FMA_DK={FMA_DK}: commit {steps - bad}/{steps} step bit a bit; copia del kernel FLA == libreria in {steps - copy_bad}/{steps}")
    return bad == 0


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    ok = False
    for FMA_DK in (True, False):
        ok |= main(steps, FMA_DK)
    sys.exit(0 if ok else 1)
