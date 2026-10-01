#!/bin/bash
# Copia sull'EBS root (/opt/llmunity-models, 230 GB liberi, nessun acquisto) cio' che serve a K3, perche' l'instance
# store /mnt/llmunity-models si cancella a ogni stop della macchina. ~135 GB: modello 126 GB, venv 7,1 GB, fork +
# worktree + estensioni ~1,1 GB, script e template. Si puo' rilanciare: rsync copia solo le differenze.
# Alla fine, se la copia coincide con l'originale, cambia /etc/fstab: dal prossimo avvio /mnt/llmunity-models e' la
# copia sull'EBS (bind mount), quindi tutti i percorsi assoluti (venv, worktree, viste, unit) restano validi.
set -euo pipefail
SRC=/mnt/llmunity-models
DST=/opt/llmunity-models
LIST="d0xin-fp8ple d0xin-fp8ple-ramview vllm-venv vllm-host-file-gather-src vllm-fase2g fase2-build prod
      prod-tests/chat-template-qwen3.8-unsloth.jinja ensure_d0xin_ple_ram.py backend-results/d0xin-ple-tmpfs-stage.json"
sudo mkdir -p $DST && sudo chown ec2-user:ec2-user $DST
df -h / | tail -1
for p in $LIST; do
  echo "[$(date -u +%T)] copio $p"
  # -a conserva symlink, permessi e date
  rsync -a --relative "$SRC/./$p" "$DST/"
done
du -sh $DST
# verifica: un secondo passaggio a secco con checksum non deve trovare differenze
DIFF=$(for p in $LIST; do rsync -a --relative --dry-run --checksum --itemize-changes "$SRC/./$p" "$DST/"; done)
[ -z "$DIFF" ] || { echo "COPIA_DIVERSA:"; echo "$DIFF" | head; exit 1; }
echo "[$(date -u +%T)] copia verificata"
if ! grep -q "^$DST $SRC none bind" /etc/fstab; then
  sudo cp -n /etc/fstab /etc/fstab.bak-k3
  # la riga dell'instance store diventa un commento, il bind mount dell'EBS prende il suo posto
  sudo sed -i "s#^\(UUID=[^ ]* $SRC xfs .*\)#\# instance store, si cancella allo stop: \1#" /etc/fstab
  echo "$DST $SRC none bind,nofail 0 0" | sudo tee -a /etc/fstab
fi
sudo findmnt --verify --tab-file /etc/fstab
echo "[$(date -u +%T)] SALVA_FINE"
