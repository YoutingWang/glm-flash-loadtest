#!/usr/bin/env bash
# run_agent_c64.sh — the coding-agent test at 16 / 32 / 64 requests in flight, exactly as we ran it for the published numbers.
#
#   bash run_agent_c64.sh                                   # both backends of https://glm-flash.aitist.ai, one after the other (~30 min)
#   URL=https://<endpoint> MODELS="<model-a> <model-b>" bash run_agent_c64.sh
#
# Uses coding-agent-serving-bench (https://github.com/zhongtian-ai/coding-agent-serving-bench) pinned to one commit, so
# everyone runs the same corpus (sha256 6e93e8a5f3f6...) and the same load generator. Needs Python >= 3.9 and git.
# Writes agent-results/<model>/ (summary.json, per-request records, SHA256SUMS) and agent-results/compare.md.
set -euo pipefail
URL=${URL:-https://glm-flash.aitist.ai}
MODELS=${MODELS:-"glm-official glm-zt"}      # glm-zt = optimized build, glm-official = unmodified official SGLang (baseline)
CONC=${CONC:-16,32,64}; WARMUP=${WARMUP:-60}; DUR=${DUR:-180}
BENCH_REF=${BENCH_REF:-78d9cc3e14ff}           # coding-agent-serving-bench commit used for our numbers
HERE=$(cd "$(dirname "$0")" && pwd); B=$HERE/.agent-bench; OUT=${OUT:-$PWD/agent-results}

if [ ! -d "$B/.git" ]; then git clone -q https://github.com/zhongtian-ai/coding-agent-serving-bench "$B"; fi
git -C "$B" fetch -q origin && git -C "$B" checkout -q "$BENCH_REF"
python3 -c 'import aiohttp' 2>/dev/null || { echo "pip install aiohttp  (the only dependency)"; exit 1; }
mkdir -p "$OUT"
for M in $MODELS; do
  echo "== $M @ $URL  concurrency $CONC, warm-up ${WARMUP}s, measured ${DUR}s per level"
  python3 "$B/bench/run.py" --url "$URL" --model "$M" --label "$M" --out "$OUT/$M" \
    --concurrency "$CONC" --warmup "$WARMUP" --duration "$DUR" --hardware "${HARDWARE:-}" ${RARGS:-}
done
python3 "$B/bench/compare.py" $(for M in $MODELS; do echo "$OUT/$M"; done) --md "$OUT/compare.md"
echo "table: $OUT/compare.md"
