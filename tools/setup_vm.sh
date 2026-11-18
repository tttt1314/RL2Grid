#!/bin/bash
# One-shot, idempotent setup of a fresh Ubuntu VM (tested on GCE c2d-highcpu-32, Ubuntu 24.04) for this fork.
#
#   curl -sSL https://raw.githubusercontent.com/tttt1314/RL2Grid/dev/fixed-async/tools/setup_vm.sh | bash
#   # or, from a checkout:  bash tools/setup_vm.sh
#
# Env vars:
#   ENV_IDS  RL2Grid env ids whose grid2op datasets are downloaded + decompressed (default "bus14 bus36-MO-v0")
#   BRANCH   branch to check out (default dev/fixed-async)
#
# Steps: apt deps -> Miniconda -> clone fork -> conda env (conda_env.yml) -> editable install -> unzip action
# spaces -> download datasets -> decompress chronics (*.csv.bz2 -> *.csv, originals kept; grid2op prefers .csv,
# which makes every env reset ~2x faster) -> ~/logs, ~/queue.
set -euo pipefail
ENV_IDS=${ENV_IDS:-"bus14 bus36-MO-v0"}
BRANCH=${BRANCH:-dev/fixed-async}
REPO=https://github.com/tttt1314/RL2Grid.git
DIR=$HOME/RL2Grid-fork
log() { echo -e "\n\033[1;34m[setup]\033[0m $*"; }

log "apt packages"
sudo apt-get update -qq && sudo apt-get install -y -qq git unzip bzip2 build-essential htop >/dev/null

if [ ! -d "$HOME/miniconda3" ]; then
  log "Miniconda"
  curl -sSLo /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda config --set always_yes yes >/dev/null
for ch in https://repo.anaconda.com/pkgs/main https://repo.anaconda.com/pkgs/r; do
  conda tos accept --override-channels --channel "$ch" >/dev/null 2>&1 || true
done

log "clone $REPO ($BRANCH)"
if [ -d "$DIR/.git" ]; then git -C "$DIR" fetch -q origin && git -C "$DIR" checkout -q "$BRANCH" && git -C "$DIR" pull -q
else git clone -q -b "$BRANCH" "$REPO" "$DIR"; fi
cd "$DIR"

if ! conda env list | grep -q '^rl2grid '; then
  log "conda env rl2grid"
  conda env create -q -f conda_env.yml
fi
conda activate rl2grid
log "editable install (main.py and the queue always use this checkout, never a stale site-packages copy)"
pip uninstall -y -q rl2grid >/dev/null 2>&1 || true
pip install -q -e . py-spy

log "action spaces"
(cd env && unzip -oq action_spaces.zip)

log "datasets: $ENV_IDS"
python - "$ENV_IDS" <<'EOF'
import json, sys, warnings
warnings.filterwarnings("ignore")
import grid2op
cfg = json.load(open("env/scenario.json"))["environments"]
for env_id in sys.argv[1].split():
    gid = cfg[env_id]["grid2op_id"]
    print(f"  {env_id} -> {gid}", flush=True)
    grid2op.make(gid).close()          # downloads on first use
EOF

log "decompress chronics (keeps the .bz2 originals)"
python - "$ENV_IDS" <<'EOF' | while read -r d; do
import json, sys
cfg = json.load(open("env/scenario.json"))["environments"]
print("\n".join(cfg[e]["grid2op_id"] for e in sys.argv[1].split()))
EOF
  C="$HOME/data_grid2op/$d/chronics"
  [ -d "$C" ] || { echo "  $d: no chronics dir, skipped"; continue; }
  n=$(find "$C" -name '*.csv.bz2' | while read -r f; do [ -e "${f%.bz2}" ] || echo "$f"; done | tee /tmp/todo_$$ | wc -l)
  if [ "$n" -gt 0 ]; then xargs -a /tmp/todo_$$ -P "$(nproc)" -n 20 bunzip2 -k -f; fi
  echo "  $d: decompressed $n files, chronics now $(du -sh "$C" | cut -f1)"
done
rm -f /tmp/todo_$$

mkdir -p "$HOME/logs" "$HOME/queue"
log "done: $(git log --oneline -1)"
