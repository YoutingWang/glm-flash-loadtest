# glm-flash-loadtest

A small, dependency-light **closed-loop load test for OpenAI-compatible chat endpoints**, built to answer one practical
question: *how fast does this deployment serve a low-concurrency, long-prompt, short-answer workload, and is a
comparison between two deployments fair?*

- 5 requests in flight, a fixed 200-prompt corpus (average ~5.4K tokens in / ~160 out) cycled for a fixed time
- run under **three prefix-cache states**, because cache hit rate alone can change throughput by 2×
- reports throughput and p50 / p95 / p99 latency with the same definitions as common serving benchmarks
- one command to run, one command to compare several runs in a single table

It measures **speed only, not answer quality**. It is model- and engine-agnostic: anything that serves
`/v1/chat/completions` with streaming and `usage` works. The default model name is `zai-org/GLM-5.3-Flash`; change it with `MODEL=`.

## 1. The workload

**Closed loop, concurrency 5.** Five workers each send a request, wait for the stream to finish, then send the next.
This is a different regime from "hundreds of chat users": it looks like tool-using agents, retrieval QA, code reading
and document summarization — long input, short output, a handful of requests at a time.

The shape was chosen to match a published reference result sheet (5 concurrent, 200 prompts cycled for 30 minutes,
average 5,313 tokens in / 161 out). The reference corpus itself was not available, so `gen_corpus.py` builds a
synthetic stand-in:

| Property | Reference sheet | This corpus (measured server-side) |
|---|---|---|
| Distinct prompts, cycled | 200 | 200 |
| Requests in flight | 5 (closed loop) | 5 (closed loop) |
| Average input tokens | 5,313 | 5,440–5,470 |
| Average output tokens | 161 | 160–164 |
| Output length p50 / p95 / p99 | ≈93 / 520 / 1,030 (back-computed from latencies) | 98 / 508 / 1,804 |

Corpus mix (the **length distribution is fitted to the reference; the task mix is an assumption**): 80 multi-turn
tool-using agent conversations (tool schemas in the prompt, short replies), 70 long-context retrieval questions,
30 code-reading tasks, 20 long-document summaries; Chinese and English. Text is Wikipedia (CC BY-SA 4.0) and
open-source Python source (Apache-2.0) wrapped in generated task prompts. Each request carries a `max_tokens` cap drawn from a
log-normal distribution fitted to the reference latency percentiles. Prompts of one simulated "app" share a system
prompt, as real applications do. Thinking is switched off through `chat_template_kwargs` unless you pass `--thinking`.

`corpus.jsonl.gz` is the corpus used for every number below; use it as is. `gen_corpus.py` rebuilds a similar one
(fixed seed, but it fetches live Wikipedia pages and needs a tokenizer file, so the result will not be byte-identical).

## 2. Three cache states — the part most comparisons get wrong

A serving engine with a prefix cache does not recompute the beginning of a prompt it has already seen. The same server can
look 2× faster or slower depending on how much of each prompt is already cached, so **always state the cache state, and
measure more than one**. A reference sheet that only says the cache was "warm" does not tell you the hit rate, so we bracket it:

| `--mode` | What it does | Input-token cache hit | Meaning |
|---|---|---|---|
| `replay` (use `--warm-pass`) | The 200 prompts are cycled unchanged after one warm-up pass | ≈97% | Upper bound: every prompt has been seen |
| `sysshared` | A unique marker is inserted at the start of the first user turn | ≈20% | Only the system prompt and tool list can hit (shared by requests of one app) |
| `fresh` | A unique marker is inserted at the very start | ≈15% | Lower bound: almost everything is recomputed (a little still hits because the chat template puts tool definitions before the marker) |

Real traffic usually sits somewhere between. Report all three and let the reader pick.

## 3. Metrics

| Metric | Definition |
|---|---|
| **Output tok/s** | Total output tokens across all requests ÷ wall-clock time. The headline number. Output tokens come from the server's `usage.completion_tokens`, not from counting stream chunks |
| **TTFT** | Time from sending the request to the first streamed delta (content, reasoning or tool call) |
| **TPOT** | (end-to-end time − TTFT) ÷ (output tokens − 1) |
| **ITL** | Gap between consecutive streamed chunks. Some servers put several tokens in one chunk, so ITL is not time per token; use its p99 as a "stall" indicator rather than comparing medians across servers |
| **E2E** | Whole-request latency |
| p50 / p95 / p99 | Median / slowest 5% / slowest 1% |

Also recorded: cache hit fraction (`usage.prompt_tokens_details.cached_tokens`), average tokens in/out, the fraction of
requests that stopped because they hit `max_tokens`, and failures. Failed requests are counted and reported, never dropped silently.

## 3.1 Quick start

```bash
pip install aiohttp

# all three cache states against any OpenAI-compatible endpoint; 3 x 400 s plus a warm-up pass, ~22 minutes
URL=http://localhost:30000 NAME=my-setup DUR=400 bash run_bench.sh

# run several setups, then compare them in one table
python3 summarize.py results/*
```

Environment variables for `run_bench.sh`: `URL`, `NAME` (required; results go to `results/$NAME/`), `DUR` (seconds per
state; use 1000 for numbers you publish), `CONC` (default 5), `MODEL`, `API_KEY` (sent as `Authorization: Bearer`),
`NO_TEMPLATE_KWARGS=1` (for servers that reject `chat_template_kwargs`), `NOTE` (who ran it, on what; stored in the summary), `RARGS` (extra arguments for `replay.py`, e.g. `--temperature 0`).

One state only (run `gunzip -k corpus.jsonl.gz` first):
`python3 replay.py --corpus corpus.jsonl --out out --mode fresh --duration 400`.
Each run writes `requests.jsonl` (per request) and `summary.json`.

## 3.2 Provenance and exclusivity: who ran this, when, and was the server ours alone?

A throughput number is only meaningful if nobody else was loading the server while it was measured, and if someone else can
tell that. Every run therefore writes a `provenance` block into `summary.json`:

| Recorded | Why |
|---|---|
| `start_utc`, `end_utc`, and a UTC `ts` on every request in `requests.jsonl` | When it ran; lets a server operator line it up with their own logs |
| `run_id` (random, or `--run-id`), sent as the `X-Loadtest-Run-Id` header and as the client's `User-Agent` | Attributes requests in server / proxy logs to this run |
| `corpus_sha256`, `replay_py_sha256`, `repo_commit`, `argv`, Python / aiohttp versions, hashed client hostname | Others can verify they ran the same corpus and the same code with the same arguments |
| `requests_jsonl_sha256` | `summarize.py` recomputes it; a hand-edited result shows `modified!` |
| `server_*.json` next to the results (`/v1/models`, plus engine info endpoints if present) and their hashes | What the server said it was. These files can contain engine settings; look before you publish them |
| `--note` | Free text: who ran it, on what |

**Exclusivity is checked, not assumed.** If the server exposes Prometheus `/metrics` (SGLang, vLLM and compatible engines do):

1. **Idle check before the start.** If the server already has running or queued requests, the run aborts (`--allow-busy` records it as contaminated instead).
2. **Sampler during the run (1 Hz).** More running requests than `--concurrency` means someone else is sending traffic.
3. **Token accounting.** Tokens the server generated over the whole run (warm-up included) are compared with the tokens we received. Ratio above 1.05 means the server did work we did not ask for.

The verdict is `clean`, `foreign_traffic_suspected` (with the reason, peak and counts) or `unverified` (no `/metrics`: then the
result is only as exclusive as the tester says, so run it on an isolated server and say so in `--note`). `summarize.py` prints the verdict and start time on
every row. **Do not compare or publish rows that are not `clean`** without saying so.

**Cache isolation.** Unless you pass `--no-run-salt`, the run id is mixed into the first message of every prompt, so cache hits can only come from
this run's own warm-up pass, never from an earlier run or another client using the same prompts. The cache hit fractions stay as described in section 2.
A side effect: two runs of the same setup never share cache, so repeat runs are independent.

**Unique and attributable.** A run is identified by `run_id` + start time + corpus / script hashes. To publish a result, publish
`summary.json` and `requests.jsonl` together with the server's launch command; others can then rerun with the same hashes and compare.
Server-side, the unsalted comparison "did anyone else use my server" is only as good as `/metrics`; for a hard guarantee,
run on a dedicated machine or behind a firewall that only allows the load generator.

## 3.3 High concurrency (64 and more) against a hosted endpoint

The same tool does a concurrency sweep; only `--concurrency` and the connection details change.

```bash
export API_KEY=<key you were given>            # sent as "Authorization: Bearer ..."
# one run, 64 requests in flight, 5 minutes, almost no cache hits (the harder case)
python3 replay.py --corpus corpus.jsonl --out out-c64 --url https://<endpoint> --model <model-or-backend-id> \
    --concurrency 64 --duration 300 --mode fresh --note "who/where"

# or the whole matrix: three cache states at each concurrency, one table
for C in 16 64 128; do
  URL=https://<endpoint> MODEL=<model-or-backend-id> NAME=c$C CONC=$C DUR=300 bash run_bench.sh
done
python3 summarize.py results/c*
```

`--url` is the base without `/v1` (the tool appends `/v1/chat/completions`). Things that matter at high concurrency:

- **Read TTFT and p99, not only tok/s.** With 64 in flight, aggregate tok/s rises but each request gets slower; the queueing shows up in TTFT p95/p99 and in E2E. Report both.
- **Closed loop means the server is never overloaded beyond `--concurrency`.** If you also want to see the rejection behaviour, raise the concurrency until you see `failed > 0`; HTTP 429 means the endpoint's own in-flight cap, not a model failure. Failed requests are listed in `summary.json` (`errors`) and never dropped.
- **Behind a proxy there is no `/metrics`**, so the exclusivity verdict is `unverified`. Say in `--note` whether you were the only user, and do not run two load tests against the same endpoint at the same time: they slow each other down and neither number is valid.
- **The corpus is ~5.4K tokens per prompt.** 64 in flight is ~350K tokens of prompt on the server at once; make sure the server's context / KV-cache capacity is large enough, otherwise requests queue and TTFT grows without any engine problem.
- **Warm-up**: `--mode replay` needs `--warm-pass` (the 200 prompts at 64 concurrency take about a minute). `fresh` and `sysshared` need none.
- A 60-second run is only a connectivity check; tails need 300 s or more.

## 3.4 The coding-agent test at 16 / 32 / 64 (the numbers we publish for GLM-5.3-Flash)

The tests above use 5.4K-token prompts. Coding agents send much longer, growing prompts, and "does it hold 64 agents at
once" is a separate question, so it has its own test in `agent/`.

- **Workload:** 120 agent sessions (2,633 requests), a median prompt of ~19K tokens (p90 41K), and 6 tool schemas. Consecutive turns share a growing prefix, as real agents do.
- **Load:** closed loop, so exactly N requests are always in flight. Each level gets a 60 s warm-up, then is measured for 180 s.
- **Fixed output:** fixed output lengths (`ignore_eos`) mean every deployment does the same work.
- **Built-in checks:** exact output length, canary accuracy under load, temperature-0 fingerprints before and after, and an implausibly fast prefill check. If you can stream the server's log in (`--server-log-cmd`), it also checks that nobody else was using the server.

**This is the test behind our published GLM-5.3-Flash numbers, and everything needed to run it is in this repository:**

| Path | What |
|---|---|
| `agent/bench/run.py` | load generator (sha256 `f2abcfcbe416…`) |
| `agent/bench/compare.py` | turns several runs into one table |
| `agent/corpus/coding_v1.jsonl.gz` | the request corpus (sha256 `6e93e8a5f3f6…`) |
| `agent/results/2026-11-12-glm-flash-l20x/`, `agent/results/2026-11-12-glm-flash-l20x-c1-8/` | our raw results (16/32/64 and 1/4/8): summary, every request, canaries, `SHA256SUMS` |
| `run_agent_c64.sh` | one command that checks the hashes and runs the test |

`agent/` is a byte-identical copy of the load generator and corpus from coding-agent-serving-bench, commit `78d9cc3e14ff`
(Apache-2.0, see `agent/LICENSE`). The corpus contains excerpts of CPython source, PSF License 2.0.

**Check that our published numbers came from this exact code.** Every `summary.json` records the sha256 of the load
generator and the corpus that produced it:

```bash
shasum -a 256 agent/bench/run.py agent/corpus/coding_v1.jsonl.gz
grep -h -o '"\(script\|corpus\)_sha256": "[0-9a-f]*"' agent/results/*/*/summary.json | sort | uniq -c
```

Both hashes appear in all six of our result sets. `run_agent_c64.sh` refuses to run if the local files differ.

**Reproduce:**

```bash
pip install aiohttp
bash run_agent_c64.sh                    # https://glm-flash.aitist.ai, glm-official then glm-zt, ~30 min, one table
CONC=1,4,8 bash run_agent_c64.sh         # the low-concurrency levels, same script
python3 agent/bench/compare.py agent-results/glm-official agent-results/glm-zt agent/results/2026-11-12-glm-flash-l20x/*-https
```

On `https://glm-flash.aitist.ai`, the model name selects the backend: `glm-zt` is the optimized build and `glm-official` is
unmodified official SGLang. Both run on identical 8× L20X 141 GB nodes with the same weights and chat template. No API key
is needed, and up to 64 requests may be in flight per client IP. The 5-in-flight test above (`replay.py` / `run_bench.sh`) is a
different workload, and its numbers are not comparable with these.

**Our results** (2026-11-12, same command from a workstation on the public internet, 8× L20X per backend):

| Concurrency | Output tok/s: glm-official → **glm-zt** | TTFT p50 / p99 (ms): glm-official → **glm-zt** | TPOT p50 (ms): official → zt |
|---:|---|---|---|
| 16 | 972 → **1,619 (1.67×)** | 561 / 795 → **580 / 757** | 14.1 → 7.7 |
| 32 | 1,469 → **2,418 (1.65×)** | 582 / 804 → **604 / 861** | 19.1 → 10.8 |
| 64 | 2,112 → **3,297 (1.56×)** | 603 / 900 → **650 / 1,054** | 27.1 → 16.3 |

Low concurrency, same script (`CONC=1,4,8`), same day, same path:

| Concurrency | Output tok/s: glm-official → **glm-zt** | TTFT p50 (ms): official → zt | TPOT p50 (ms): official → zt |
|---:|---|---|---|
| 1 | 99 → **156 (1.57×)** | 666 → 686 | 7.6 → 4.0 |
| 4 | 374 → **583 (1.56×)** | 532 → 562 | 8.7 → 4.8 |
| 8 | 636 → **1,005 (1.58×)** | 542 → 562 | 10.6 → 6.0 |

All integrity checks passed on every level of all runs, with 0 failed requests. Our server logs were streamed in, so the exclusivity check passed.
One exception is worth knowing about. During the last ~1 min of glm-zt's 8-agent level, one request from another user (25K output tokens) was also running.
The exclusivity rule tolerates one extra request, so the check still passed. That overlap can only have lowered glm-zt's number; the server's "idle after" check recorded it.
With the client on the server node itself (`official-sglang`, `a1-l20x`), the same backends reach 2,148 → 3,357 tok/s at 64. The
public path adds a flat ~0.45 s to time to first token and costs 2–9% of throughput.

What to watch for when you reproduce:

- **Pin the backend** with `glm-zt` / `glm-official`. The generic name `glm-5.3-flash` falls back from the optimized to the official backend if the first is unavailable, so a run could mix both.
- **Run one test at a time,** and not at the same time as someone else. Without the server log you cannot prove exclusivity yourself (the verdict is `not checked`), so say in your notes whether you were alone.
- **Expect your TTFT to differ from ours** by your network round trip to Cloudflare. Throughput should be close.
- **The first minute after a server restart is slow** (JIT compilation). The 60 s warm-up covers this.

## 4. Example results

`example-results/` holds raw `summary.json` files from three deployments of the same model, measured with this tool
(5 in flight, 400 s per state, 0 failed requests in every run):

| Setup | Cache hit | Output tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Avg out tokens | Hit `max_tokens` |
|---|---|---|---|---|---|---|
| A | 97% | 483 | 56 / 229 | 9.5 | 163 | 59% |
| A | 20% | 375 | 175 / 515 | 11.4 | 161 | 58% |
| A | 15% | 370 | 180 / 514 | 11.7 | 163 | 60% |
| B | 97% | 842 | 68 / 230 | 5.3 | 163 | 59% |
| B | 20% | 619 | 149 / 430 | 6.7 | 164 | 59% |
| B | 15% | 610 | 154 / 410 | 6.8 | 163 | 59% |
| C | 97% | 772 | 110 / 253 | 5.6 | 208 | 85% |
| C | 20% | 422 | 780 / 3185 | 5.8 | 209 | 86% |
| C | 15% | 402 | 891 / 3359 | 5.8 | 209 | 87% |

`python3 summarize.py example-results/*` prints the full table. These three predate the provenance block (no timestamps, exclusivity unverified, no run salt), so treat them as illustration, not as reproducible evidence; new runs record everything in section 3.2. Setups A and B ran on the same hardware type with the
same weights, corpus and chat template, which is what makes them comparable. **Setup C is not comparable** to A and B: its chat template could
not switch thinking off, so most of its output is reasoning (average 208 tokens, 85% of requests ran into `max_tokens`) and its TTFT includes
thinking time. That is exactly the kind of difference the checklist below exists to catch.

## 5. Checklist for a fair comparison

1. **Same chat template on both sides.** If one server's template cannot turn thinking off, it will reason until `max_tokens`
   on almost every request, emit no tool calls, and the numbers stop meaning anything. (We hit this ourselves: with the
   model's stock template 99% of requests ran to the length cap.) The template only changes the request format; it is not a speed optimization.
2. **Both runs `clean`** in the "exclusive?" column, with start times that do not overlap on a shared server (section 3.2).
3. **Same corpus, same seed, same `--mode`, same concurrency, same duration.** `replay.py` fixes the order with `--seed 1`.
4. **Check that your baseline reproduces the reference.** Compare average output tokens (163 vs. 161 here) and the median
   inter-chunk gap (9.24 ms vs. 9.24 ms here). If those do not line up, you are not measuring the same workload yet.
5. **Look at the `hit max_tokens` column.** It should be similar across setups. A big difference means the outputs differ in kind, not just in speed.
6. **Report the cache hit column**, and all three cache states.
7. **Noise.** Repeating a run gives ±1% on throughput and medians; tail metrics (p99) from 5-minute runs vary by up to ±25%. Use `DUR=1000` for anything you publish.
8. **Tail latency is where cache states differ most.** With few cache hits, a running request gets interrupted by other requests' prompt processing, which shows up in ITL p99 and TTFT p99.
9. **Do not mix sources.** A reference number copied from someone's result sheet was not measured on your machine at the same time; say so, and prefer running both sides yourself.

## 6. What this test does not cover

High concurrency (dozens to hundreds of long sessions), much longer inputs, answer quality, and sampling settings other
than the server default (requests carry no `temperature` unless you pass one).

## Files

| File | Purpose |
|---|---|
| `provenance.py` | Run metadata, hashes, and the exclusivity checks (idle check, sampler, token accounting) |
| `replay.py` | Load generator: closed-loop replay, three cache modes, TTFT / TPOT / ITL / E2E percentiles |
| `run_bench.sh` | Runs the three cache states and prints a table |
| `summarize.py` | Merges `summary.json` files from several setups into one Markdown table |
| `run_agent_c64.sh`, `agent/` | Coding-agent test at 16 / 32 / 64 in flight (section 3.4): load generator, corpus, our raw results |
| `corpus.jsonl.gz` | The 200-prompt corpus used for every number above |
| `gen_corpus.py` | Rebuilds a similar corpus (needs network access and a tokenizer file) |
| `example-results/` | Raw summaries from three example deployments |

## License

Code and documentation: Apache-2.0. The corpus contains Wikipedia text (CC BY-SA 4.0) and Apache-2.0 source files;
reuse of the corpus must respect those licenses.
