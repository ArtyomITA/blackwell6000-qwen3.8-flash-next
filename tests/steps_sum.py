# Riassunto di un CSV della patch B.3: per (full_cg, num_reqs) numero di step e mediane/p90 di target, draft, periodo.
# Periodi > 1 s (server fermo tra due richieste) esclusi. Uso: steps_sum.py <csv> [t0 t1]
import statistics, sys
from collections import defaultdict

t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0
t1 = float(sys.argv[3]) if len(sys.argv) > 3 else 1e20
g = defaultdict(list)
for line in open(sys.argv[1]):
    t, cg, ntok, nreq, fw, dr, per = line.strip().split(",")
    if t0 <= float(t) <= t1 and float(per) < 1000:
        g[(cg, int(nreq))].append((int(ntok), float(fw), float(dr), float(per)))


def p(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


print("full_cg,num_reqs,steps,tok_med,fwd_med,fwd_p90,draft_med,period_med,period_p90,gap_med")
for (cg, nr), rows in sorted(g.items()):
    fw = [r[1] for r in rows]; dr = [r[2] for r in rows]; per = [r[3] for r in rows]
    gap = [r[3] - r[1] - r[2] for r in rows]
    print(f"{cg},{nr},{len(rows)},{statistics.median(r[0] for r in rows):.0f},{statistics.median(fw):.2f},{p(fw, .9):.2f},"
          f"{statistics.median(dr):.2f},{statistics.median(per):.2f},{p(per, .9):.2f},{statistics.median(gap):.2f}")
