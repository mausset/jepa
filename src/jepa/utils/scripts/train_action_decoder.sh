#!/usr/bin/env bash
# Usage:
#   bash src/utils/scripts/train_action_decoder.sh \
#     -e experiments/beta-gamma-cross-4-fixed \
#     -c configs/envs/push.yaml \
#     -o results/beta-gamma-cross-4-fixed.csv \
#     [-l out.log] [-p 'checkpoint.pth']

set -Eeuo pipefail

EXPERIMENT=""
CONFIG=""
CSV=""
LOG="out.log"
PATTERN='checkpoint.pth'

usage() {
  echo "Usage: $0 -e EXP_DIR -c CONFIG_YAML -o OUTPUT_CSV [-l LOG] [-p GLOB]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  -e | --experiment)
    EXPERIMENT="$2"
    shift 2
    ;;
  -c | --config)
    CONFIG="$2"
    shift 2
    ;;
  -o | --csv | --out)
    CSV="$2"
    shift 2
    ;;
  -l | --log)
    LOG="$2"
    shift 2
    ;;
  -p | --pattern)
    PATTERN="$2"
    shift 2
    ;;
  -h | --help) usage ;;
  *)
    echo "Unknown arg: $1"
    usage
    ;;
  esac
done

[[ -z "$EXPERIMENT" || -z "$CONFIG" || -z "$CSV" ]] && usage
[[ -d "$EXPERIMENT" ]] || {
  echo "No such directory: $EXPERIMENT"
  exit 2
}

mkdir -p "$(dirname "$CSV")"
: >"$LOG"
echo "lambda, train_mse, val_mse" >"$CSV"

# Collect checkpoints first for clear count and to avoid subshell issues
mapfile -d '' -t CKPTS < <(find "$EXPERIMENT" -type f -name "$PATTERN" -print0)
COUNT=${#CKPTS[@]}
echo "Found $COUNT checkpoints matching '$PATTERN' under $EXPERIMENT"

if ((COUNT == 0)); then
  echo "Nothing to do."
  exit 0
fi

for CKPT in "${CKPTS[@]}"; do
  printf '===== %s | %s =====\n' "$(date -Iseconds)" "$CKPT" >>"$LOG"
  # Keep stderr on terminal so tqdm is visible; only stdout is parsed.
  if LINE=$(python src/factored_ssl/planning/train_action_decoder.py --config "$CONFIG" --checkpoint "$CKPT" | tail -n 1); then
    echo "$LINE" >>"$CSV"
    echo "$LINE" >>"$LOG"
  else
    echo "ERROR running on $CKPT" >>"$LOG"
  fi
done

ROWS=$(($(wc -l <"$CSV") - 1))
echo "Wrote $ROWS rows to $CSV"
