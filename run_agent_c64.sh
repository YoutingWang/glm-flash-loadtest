#!/usr/bin/env bash
# run_agent_c64.sh — the coding-agent test at 16 / 32 / 64 requests in flight, exactly as we ran it for the published numbers.
#
#   bash run_agent_c64.sh                                   # both backends of https://glm-flash.aitist.ai, one after the other (~30 min)
#   URL=https://<endpoint> MODELS="<model-a> <model-b>" bash run_agent_c64.sh
#
# Self-contained: the load generator and corpus are in agent/ (a byte-identical copy of coding-agent-serving-bench
# commit 78d9cc3e14ff, Apache-2.0). Before running, the script checks their sha256 against the hashes recorded in our
# published results (agent/results/*/*/summary.json -> manifest.script_sha256 / corpus_sha256).
# Needs Python >= 3.9 and aiohttp. Writes agent-results/<model>/ and agent-results/compare.md.
set -euo pipefail
URL=${URL:-https://glm-flash.aitist.ai}
MODELS=${MODELS:-"glm-official glm-zt"}      # glm-zt = optimized build, glm-official = unmodified official SGLang (baseline)
CONC=${CONC:-16,32,64}; WARMUP=${WARMUP:-60}; DUR=${DUR:-180}
HERE=$(cd "$(dirname "$0")" && pwd); A=$HERE/agent; OUT=${OUT:-$PWD/agent-results}
sha() { python3 -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"; }
want_run=$(python3 -c 'import json,glob;print(json.load(open(sorted(glob.glob("'"$A"'/results/*/*/summary.json"))[0]))["manifest"]["script_sha256"])')
want_corpus=$(python3 -c 'import json,glob;print(json.load(open(sorted(glob.glob("'"$A"'/results/*/*/summary.json"))[0]))["manifest"]["corpus_sha256"])')
[ "$(sha "$A/bench/run.py")" = "$want_run" ] || { echo "agent/bench/run.py does not match the published results"; exit 1; }
[ "$(sha "$A/corpus/coding_v1.jsonl.gz")" = "$want_corpus" ] || { echo "agent/corpus does not match the published results"; exit 1; }
echo "load generator sha256 ${want_run:0:12}…  corpus sha256 ${want_corpus:0:12}…  (same as the published results)"
python3 -c 'import aiohttp' 2>/dev/null || { echo "pip install aiohttp  (the only dependency)"; exit 1; }
mkdir -p "$OUT"
for M in $MODELS; do
  echo "== $M @ $URL  concurrency $CONC, warm-up ${WARMUP}s, measured ${DUR}s per level"
  python3 "$A/bench/run.py" --url "$URL" --model "$M" --label "$M" --out "$OUT/$M" \
    --concurrency "$CONC" --warmup "$WARMUP" --duration "$DUR" --hardware "${HARDWARE:-}" ${RARGS:-}
done
python3 "$A/bench/compare.py" $(for M in $MODELS; do echo "$OUT/$M"; done) --md "$OUT/compare.md"
echo "table: $OUT/compare.md   (ours: python3 agent/bench/compare.py agent/results/2026-11-12-glm-flash-l20x/*)"
