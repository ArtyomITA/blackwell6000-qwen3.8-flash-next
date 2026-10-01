# Copertura offline per F: quota dei token GENERATI dal modello che cadono nell'insieme di bozza (ID < N, o lista
# per frequenza costruita su meta' dei dati e misurata sull'altra meta'). Solo CPU.
import collections, glob, json, sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/mnt/llmunity-models/d0xin-fp8ple-ramview")
texts = []
for p in glob.glob("/mnt/llmunity-models/backend-results/*stream*.jsonl") + glob.glob("/var/tmp/llmunity-lcb32/lcb_vllm-*.jsonl"):
    for line in open(p, errors="ignore"):
        try:
            j = json.loads(line)
        except Exception:
            continue
        stack = [j]
        while stack:  # tutte le stringhe lunghe del record (contenuto, ragionamento, codice)
            x = stack.pop()
            if isinstance(x, dict): stack += list(x.values())
            elif isinstance(x, list): stack += x
            elif isinstance(x, str) and len(x) > 200: texts.append(x)
ids = [tok.encode(t, add_special_tokens=False) for t in texts]
half = len(ids) // 2
train = collections.Counter(i for s in ids[:half] for i in s)
test = [i for s in ids[half:] for i in s]
allt = [i for s in ids for i in s]
print(f"testi {len(texts)}, token {len(allt)}")
for n in (32768, 65536, 98304, 131072):
    print(f"ID < {n}: copertura {sum(i < n for i in allt) / len(allt):.4f}")
for k in (32768, 47000, 65536):
    hot = {i for i, _ in train.most_common(k)}
    print(f"top-{k} per frequenza (meta' A) su meta' B: copertura {sum(i in hot for i in test) / max(len(test), 1):.4f}")
