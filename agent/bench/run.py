#!/usr/bin/env python3
"""run.py — fair, reproducible load test of an OpenAI-compatible LLM endpoint with a coding-agent workload.

  python3 bench/run.py --url http://HOST:PORT --model MODEL --label my-setup --out results/my-setup \
      --concurrency 16,32,64 --warmup 60 --duration 180 [--server-log-cmd "sudo podman logs -f --since 1s glm"]

What it does, per concurrency level N (closed loop: always exactly N requests in flight):
  * N simulated agents each replay a coding session from the corpus turn by turn (request k = conversation up to
    step k, so a session's requests share a growing prefix, like a real agent). Every session instance gets a fresh
    random repository path at the top of its task, so nothing can be served from a cache of an earlier run, while
    within-session prefix caching (which real agents get) still works. All agents share one system prompt + tools.
  * Every request asks for a fixed number of output tokens (`ignore_eos`), so every setup generates exactly the same
    amount of text; requests where the server returns a different count are flagged.
  * The first --warmup seconds are not measured. Throughput counts only tokens streamed inside the measurement window.

Integrity checks (results go to integrity.json and are printed; a FAIL means the comparison is not fair):
  exclusive   Was our traffic the only traffic? Before the run we require the server to be idle; during the run we
              compare the server's own count of running+queued requests (from its log, --server-log-cmd; SGLang and
              vLLM log formats are parsed) with the number of requests we had in flight at that moment, and the
              server-side prompt-token count with ours. Any extra load = someone else is using the server.
              Without server logs we fall back to an idle single-request probe before and after (weaker; marked).
  fixed_len   completion_tokens == requested length for >= 99% of requests (server honoured ignore_eos, did not
              truncate or pad); characters-per-token within a plausible band (token counts are not inflated).
  canary      ~2% of requests are short questions with a computable answer (what does this Python expression print),
              asked in the same agent context at temperature 0, interleaved with the load. Accuracy under load vs idle
              catches a server that degrades quality (or swaps models) when busy.
  fingerprint Temperature-0 answers to fixed prompts at the start and the end must match (same model throughout).
  prefill     No request may report an implausible uncached-prefill speed (would indicate replayed/cached responses).
"""
import argparse, asyncio, gzip, hashlib, json, os, platform, random, re, shlex, statistics, sys, time, uuid
from datetime import datetime, timezone
import aiohttp

VERSION = '1.0.0'
ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
ap.add_argument('--url', required=True, help='base URL, e.g. http://localhost:30000 (path /v1/chat/completions is appended)')
ap.add_argument('--model', required=True)
ap.add_argument('--label', required=True, help='name of the setup, used in reports')
ap.add_argument('--out', required=True)
ap.add_argument('--corpus', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'corpus', 'coding_v1.jsonl.gz'))
ap.add_argument('--concurrency', default='16,32,64')
ap.add_argument('--warmup', type=float, default=60)
ap.add_argument('--duration', type=float, default=180)
ap.add_argument('--temperature', type=float, default=0.6)
ap.add_argument('--top-p', type=float, default=0.95)
ap.add_argument('--thinking', action='store_true', help='send chat_template_kwargs.enable_thinking=true (default false)')
ap.add_argument('--no-tools', action='store_true', help='do not send the tool schemas (only if the server cannot accept them)')
ap.add_argument('--no-ignore-eos', action='store_true', help='let the model stop early (output lengths then differ between setups)')
ap.add_argument('--api-key-env', default=None, help='name of an environment variable holding the API key (the key itself is never logged)')
ap.add_argument('--server-log-cmd', default=None, help='shell command that streams the server log (stdout), for the exclusivity check')
ap.add_argument('--canary-frac', type=float, default=0.02)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--hardware', default='', help='free text, recorded in the manifest (e.g. "8x H200 141GB, 1 node")')
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
CONC = [int(x) for x in A.concurrency.split(',')]
KEY = os.environ.get(A.api_key_env, '') if A.api_key_env else ''
HEADERS = {'User-Agent': 'OpenAI/Python 1.54.0', 'Content-Type': 'application/json'}  # generic client header, no benchmark marker
if KEY: HEADERS['Authorization'] = f'Bearer {KEY}'

# ---------------- corpus ----------------
corpus_sha = hashlib.sha256(open(A.corpus, 'rb').read()).hexdigest()
with gzip.open(A.corpus, 'rt', encoding='utf-8') as f:
    meta = json.loads(f.readline())['_meta']
    SESSIONS = [json.loads(l) for l in f]
TOOLS = meta['tools']
SYSTEM = SESSIONS[0]['messages'][0]['content']
RNG = random.Random(A.seed)
ORDER = list(range(len(SESSIONS))); RNG.shuffle(ORDER)
WORDS = ['atlas', 'birch', 'cobalt', 'delta', 'ember', 'fjord', 'garnet', 'harbor', 'iris', 'juniper', 'kepler', 'lumen', 'maple', 'nimbus', 'onyx', 'pylon', 'quartz', 'raven', 'sierra', 'tundra']

def salted(messages, rng):
    m = [dict(x) for x in messages]
    repo = f'/workspace/{rng.choice(WORDS)}-{uuid.uuid4().hex[:10]}'
    m[1]['content'] = f'Repository: {repo} (branch {rng.choice(["main", "dev", "fix"])}-{uuid.uuid4().hex[:6]})\n\n' + m[1]['content']
    return m

def body(messages, max_tokens, temperature=None, ignore_eos=None, tools=True):
    b = {'model': A.model, 'messages': messages, 'max_tokens': max_tokens, 'stream': True, 'stream_options': {'include_usage': True},
         'temperature': A.temperature if temperature is None else temperature, 'top_p': A.top_p,
         'chat_template_kwargs': {'enable_thinking': bool(A.thinking)}}
    if (not A.no_ignore_eos) if ignore_eos is None else ignore_eos: b['ignore_eos'] = True
    if tools and not A.no_tools:
        b['tools'] = TOOLS
        # tool schemas stay in the prompt (realistic prefill), but the server must not cut generation at a tool call,
        # otherwise output lengths depend on each server's tool parser instead of being identical
        if b.get('ignore_eos') or temperature == 0.0: b['tool_choice'] = 'none'
    return b

# ---------------- canaries ----------------
WORDS2 = ['parser', 'socket', 'buffer', 'token', 'cursor', 'thread', 'module', 'widget', 'vector', 'kernel', 'schema', 'router']
def canary(rng):
    """Easy questions with one computable answer (a working model gets ~100% at temperature 0, busy or idle)."""
    kind = rng.randrange(4)
    if kind == 0:
        a, b = rng.randint(3, 12), rng.randint(3, 12); expr = f'{a} * {b}'
    elif kind == 1:
        expr = repr(rng.choice(WORDS2)) + '.upper()'
    elif kind == 2:
        expr = f'sorted({rng.sample(range(1, 10), 4)})'
    else:
        expr = f'len({rng.choice(WORDS2)!r})'
    ans = str(eval(expr))
    msgs = [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': f'Before we continue with the repository, a quick check: what does `print({expr})` output in Python 3? Reply with only the printed value, nothing else.'}]
    return msgs, ans
def canary_ok(text, expected):
    t = text.strip().strip('`').strip()
    return t == expected or (t.splitlines() and t.splitlines()[-1].strip().strip('`') == expected)
FP_PROMPTS = ['Write a one-line Python expression that reverses a string s.', 'Name the Python exception raised by int("x").', 'What does git stash do? One sentence.']

# ---------------- one request ----------------
async def call(sess, b):
    t0 = time.time(); stamps = []; usage = {}; fin = None; text = []; err = None
    try:
        async with sess.post(A.url.rstrip('/') + '/v1/chat/completions', json=b) as r:
            if r.status != 200:
                return dict(t0=t0, err=f'http {r.status}: {(await r.text())[:160]}')
            async for raw in r.content:
                if not raw.startswith(b'data: {'): continue
                d = json.loads(raw[6:])
                if d.get('usage'): usage = d['usage']
                for ch in d.get('choices') or []:
                    dl = ch.get('delta') or {}
                    piece = (dl.get('content') or '') + (dl.get('reasoning_content') or '')
                    for tc in dl.get('tool_calls') or []: piece += json.dumps(tc.get('function') or {})
                    if piece: stamps.append(time.time()); text.append(piece)
                    fin = ch.get('finish_reason') or fin
    except asyncio.CancelledError:
        raise
    except Exception as e:
        err = repr(e)[:160]
    t1 = time.time()
    if err or not stamps: return dict(t0=t0, err=err or f'no content (finish={fin})')
    return dict(t0=t0, t1=t1, ttft=stamps[0] - t0, stamps=stamps, out=usage.get('completion_tokens') or 0, inp=usage.get('prompt_tokens') or 0,
                cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0, fin=fin, text=''.join(text))

# ---------------- server log monitor (exclusivity) ----------------
SG = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\] (Prefill|Decode) batch.*?#running-req: (\d+).*?#queue-req: (\d+)')
SG_TOK = re.compile(r'#new-token: (\d+), #cached-token: (\d+)')
VL = re.compile(r'(\d\d-\d\d \d\d:\d\d:\d\d).*Running: (\d+) reqs.*Waiting: (\d+) reqs')
class LogMon:
    def __init__(self, cmd): self.cmd, self.samples, self.proc, self.offsets = cmd, [], None, []
    async def start(self):
        self.proc = await asyncio.create_subprocess_shell(self.cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        asyncio.get_event_loop().create_task(self._read())
    async def _read(self):
        async for raw in self.proc.stdout:
            now = time.time(); l = raw.decode('utf-8', 'replace')
            m = SG.search(l)
            if m:
                ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
                self.offsets.append(now - ts)
                t = SG_TOK.search(l)
                self.samples.append(dict(ts=ts, kind=m.group(2), new=int(t.group(1)) if t else 0, cached=int(t.group(2)) if t else 0, running=int(m.group(3)), queue=int(m.group(4))))
                continue
            m = VL.search(l)
            if m:
                ts = datetime.strptime(f'{datetime.utcnow().year}-{m.group(1)}', '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
                self.offsets.append(now - ts)
                self.samples.append(dict(ts=ts, kind='vllm', new=0, cached=0, running=int(m.group(2)), queue=int(m.group(3))))
    def clock_offset(self):
        # log timestamps are wall-clock in the server's timezone; snap the observed lag to the nearest 15 minutes
        if not self.offsets: return 0.0
        return round(statistics.median(self.offsets) / 900) * 900
    async def stop(self):
        if self.proc and self.proc.returncode is None:
            import signal
            try: os.killpg(self.proc.pid, signal.SIGKILL)   # whole group: a child (e.g. tail) would keep the pipe open
            except ProcessLookupError: pass
            try: await asyncio.wait_for(self.proc.wait(), 5)
            except asyncio.TimeoutError: pass

IDLE_CANARY = {}
INFLIGHT = []  # (time, n_in_flight) sampled every 100 ms while a level runs
def inflight_at(t):
    lo, hi = 0, len(INFLIGHT) - 1
    if hi < 0: return 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if INFLIGHT[mid][0] <= t: lo = mid
        else: hi = mid - 1
    # max over +-1.5 s: log lines are stamped to the second and requests start/finish within that second
    return max(n for (tt, n) in INFLIGHT[max(0, lo - 15):lo + 16])

# ---------------- driver ----------------
async def run_level(n, logmon):
    res = []; canaries = []
    t_start = time.time(); t_meas = t_start + A.warmup; t_end = t_meas + A.duration
    cyc = {'i': 0}; rng = random.Random(A.seed * 1000 + n)
    async def agent(k):
        conn = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800), headers=HEADERS, connector=conn) as sess:
            while time.time() < t_end:
                s = SESSIONS[ORDER[cyc['i'] % len(SESSIONS)]]; cyc['i'] += 1
                msgs = salted(s['messages'], rng)
                for rq in s['requests']:
                    if time.time() >= t_end: return
                    if rng.random() < A.canary_frac:
                        cm, ans = canary(rng)
                        x = await call(sess, body(cm, 16, temperature=0.0, ignore_eos=False))
                        x.update(expected=ans, measured=x['t0'] >= t_meas); canaries.append(x)
                    x = await call(sess, body(msgs[:rq['n_messages']], rq['max_tokens']))
                    x.update(id=rq['id'], want=rq['max_tokens'], est_in=rq['est_input_tokens'])
                    res.append(x)
    tasks = [asyncio.ensure_future(agent(k)) for k in range(n)]
    await asyncio.sleep(max(0, t_end - time.time()))
    for t in tasks: t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    return res, canaries, t_meas, t_end

def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None

def summarize(n, res, t_meas, t_end):
    win = t_end - t_meas
    ok = [x for x in res if 'err' not in x]
    # tokens streamed inside the window: each chunk carries out/len(stamps) tokens
    out_win = sum(x['out'] / len(x['stamps']) * sum(t_meas <= s < t_end for s in x['stamps']) for x in ok if x['out'])
    done = [x for x in ok if x['t0'] >= t_meas and x['t1'] < t_end]
    started = [x for x in res if x['t0'] >= t_meas]
    failed = [x for x in started if 'err' in x]
    ms = lambda v: None if v is None else round(v * 1000, 1)
    tpot = [(x['t1'] - x['stamps'][0]) / (x['out'] - 1) for x in done if x['out'] > 1]
    itl = [b - a for x in done for a, b in zip(x['stamps'], x['stamps'][1:])]
    speed = [(x['out'] - 1) / (x['t1'] - x['stamps'][0]) for x in done if x['out'] > 1 and x['t1'] > x['stamps'][0]]
    lat = lambda xs: {f'p{p}': ms(pct(xs, p)) for p in (50, 90, 95, 99)}
    inp = sum(x['inp'] for x in done); cached = sum(x['cached'] for x in done)
    return dict(concurrency=n, window_s=win, completed=len(done), failed=len(failed),
                output_tokens_per_s=round(out_win / win, 1), requests_per_s=round(len(done) / win, 3),
                input_tokens_per_s=round(inp / win, 1), avg_input_tokens=round(inp / max(1, len(done))), avg_output_tokens=round(sum(x['out'] for x in done) / max(1, len(done))),
                cached_frac=round(cached / inp, 3) if inp else None,
                per_user_tok_s={'p50': round(pct(speed, 50) or 0, 1), 'p10': round(pct(speed, 10) or 0, 1)},
                ttft_ms=lat([x['ttft'] for x in done]), tpot_ms=lat(tpot), itl_ms=lat(itl), e2e_ms=lat([x['t1'] - x['t0'] for x in done]),
                errors=[x['err'] for x in failed[:3]])

def integrity(n, res, canaries, t_meas, t_end, logmon):
    I = {}
    ok = [x for x in res if 'err' not in x]
    if not A.no_ignore_eos and ok:
        exact = sum(x['out'] == x['want'] for x in ok) / len(ok)
        cpt = sum(len(x['text']) for x in ok) / max(1, sum(x['out'] for x in ok))
        I['fixed_len'] = dict(exact_frac=round(exact, 4), chars_per_token=round(cpt, 2), result='PASS' if exact >= 0.99 and 1.2 <= cpt <= 8 else 'FAIL')
    c = [x for x in canaries if 'err' not in x]
    if c:
        acc = round(sum(canary_ok(x['text'], x['expected']) for x in c) / len(c), 3)
        idle = IDLE_CANARY.get('accuracy')
        I['canary'] = dict(n=len(c), accuracy=acc, idle_accuracy=idle,
                           result='PASS' if idle is None or acc >= idle - max(0.05, 2 / len(c)) else 'FAIL')
    fast = [x for x in ok if x['inp'] - x['cached'] > 2000 and (x['inp'] - x['cached']) / max(1e-3, x['ttft']) > 1_000_000]
    I['prefill'] = dict(implausible=len(fast), result='PASS' if not fast else 'FAIL')
    if logmon is not None:
        off = logmon.clock_offset()
        samp = [s for s in logmon.samples if t_meas <= s['ts'] + off < t_end]
        excess = [s['running'] + s['queue'] - inflight_at(s['ts'] + off) for s in samp]
        bad = [e for e in excess if e >= 2]
        ours_new = sum(max(0, x['inp'] - x['cached']) for x in ok if t_meas <= x['t0'] < t_end)
        srv_new = sum(s['new'] for s in samp if s['kind'] == 'Prefill')
        I['exclusive'] = dict(method='server log', log_samples=len(samp), max_extra_requests=max(excess) if excess else None,
                              frac_samples_with_extra=round(len(bad) / max(1, len(samp)), 4),
                              server_prefill_tokens=srv_new, our_uncached_prompt_tokens=ours_new,
                              result='NO-DATA' if not samp else 'PASS' if len(bad) / len(samp) <= 0.01 and srv_new <= 1.15 * ours_new + 50000 else 'FAIL')
    return I

async def idle_check(logmon, secs=20):
    """Server must be idle: no running/queued requests in its log during `secs` while we send nothing."""
    if logmon is None: return dict(method='none', result='UNKNOWN')
    await asyncio.sleep(3)                       # let log lines of our own last requests arrive first
    t0 = time.time(); await asyncio.sleep(secs); await asyncio.sleep(2)
    off = logmon.clock_offset()
    win = [s for s in logmon.samples if t0 <= s['ts'] + off < t0 + secs]
    busy = [s for s in win if s['running'] + s['queue'] > 0]
    return dict(method='server log', seconds=secs, busy_log_lines=len(busy), result='PASS' if not busy else 'FAIL')

async def probe(tag):
    """Single request on an otherwise idle server: decode speed + temperature-0 fingerprints."""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), headers=HEADERS) as sess:
        s = SESSIONS[ORDER[0]]; msgs = salted(s['messages'], random.Random(7))[:s['requests'][0]['n_messages']]
        x = await call(sess, body(msgs, 256))
        fps = []
        for p in FP_PROMPTS:
            y = await call(sess, body([{'role': 'user', 'content': p}], 24, temperature=0.0, ignore_eos=False, tools=False))
            fps.append(y.get('text', y.get('err')))
        if tag == 'before':
            rng = random.Random(99); ok = 0
            for _ in range(20):
                cm, ans = canary(rng); y = await call(sess, body(cm, 16, temperature=0.0, ignore_eos=False))
                ok += 'err' not in y and canary_ok(y['text'], ans)
            IDLE_CANARY['accuracy'] = round(ok / 20, 3)
        speed = (x['out'] - 1) / (x['t1'] - x['stamps'][0]) if 'err' not in x and x['out'] > 1 else None
        return dict(tag=tag, idle_canary_accuracy=IDLE_CANARY.get('accuracy') if tag == 'before' else None, single_user_tok_s=round(speed, 1) if speed else None, ttft_ms=round(x['ttft'] * 1000, 1) if 'err' not in x else None, fingerprints=fps)

async def main():
    logmon = None
    if A.server_log_cmd:
        logmon = LogMon(A.server_log_cmd); await logmon.start(); await asyncio.sleep(3)
    man = dict(tool='coding-agent-serving-bench', version=VERSION, label=A.label, url_host=re.sub(r'//[^@/]*@', '//', A.url).split('//')[-1].split('/')[0],
               model=A.model, hardware=A.hardware, corpus=os.path.basename(A.corpus), corpus_sha256=corpus_sha,
               script_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(), args={k: v for k, v in vars(A).items() if k not in ('api_key_env',)},
               client=dict(host=platform.node(), python=platform.python_version(), aiohttp=aiohttp.__version__), started_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20), headers=HEADERS) as s:
            async with s.get(A.url.rstrip('/') + '/v1/models') as r: man['server_models'] = (await r.json()) if r.status == 200 else r.status
    except Exception as e: man['server_models'] = repr(e)[:100]
    man['idle_before'] = await idle_check(logmon)
    print(f'[{A.label}] idle check before: {man["idle_before"]["result"]}', flush=True)
    man['probe_before'] = await probe('before'); print(f'[{A.label}] idle probe: {man["probe_before"]["single_user_tok_s"]} tok/s single user', flush=True)
    levels = []
    for n in CONC:
        INFLIGHT.clear()
        st = asyncio.ensure_future(sampler_ref())
        res, can, t_meas, t_end = await run_level_tracked(n, logmon)
        st.cancel()
        S = summarize(n, res, t_meas, t_end); S['integrity'] = integrity(n, res, can, t_meas, t_end, logmon); levels.append(S)
        with open(f'{A.out}/requests_c{n}.jsonl', 'w') as f:
            for x in res: f.write(json.dumps({k: v for k, v in x.items() if k not in ('stamps', 'text')}) + '\n')
        with open(f'{A.out}/canaries_c{n}.jsonl', 'w') as f:
            for x in can: f.write(json.dumps({k: v for k, v in x.items() if k != 'stamps'}) + '\n')
        flags = ' '.join(f'{k}={v["result"]}' for k, v in S['integrity'].items())
        print(f'[{A.label}] c{n}: out {S["output_tokens_per_s"]} tok/s | TTFT p50/p99 {S["ttft_ms"]["p50"]}/{S["ttft_ms"]["p99"]} ms | '
              f'TPOT p50/p99 {S["tpot_ms"]["p50"]}/{S["tpot_ms"]["p99"]} ms | done {S["completed"]} fail {S["failed"]} | {flags}', flush=True)
        await asyncio.sleep(10)
    man['idle_after'] = await idle_check(logmon, 15)
    man['probe_after'] = await probe('after')
    man['fingerprint_match'] = man['probe_before']['fingerprints'] == man['probe_after']['fingerprints']
    man['finished_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    if logmon: await logmon.stop()
    out = dict(manifest=man, levels=levels)
    p = f'{A.out}/summary.json'; json.dump(out, open(p, 'w'), indent=1)
    with open(f'{A.out}/SHA256SUMS', 'w') as f:
        for fn in sorted(os.listdir(A.out)):
            if fn != 'SHA256SUMS': f.write(f'{hashlib.sha256(open(os.path.join(A.out, fn), "rb").read()).hexdigest()}  {fn}\n')
    print(f'[{A.label}] fingerprint match: {man["fingerprint_match"]}; idle after: {man["idle_after"]["result"]}; wrote {p}', flush=True)

# the in-flight counter is shared between the driver and the sampler
_STATE = {'inflight': 0}
async def sampler_ref():
    while True:
        INFLIGHT.append((time.time(), _STATE['inflight'])); await asyncio.sleep(0.1)
async def run_level_tracked(n, logmon):
    global call
    inner = call
    async def counted(sess, b):
        _STATE['inflight'] += 1
        try: return await inner(sess, b)
        finally: _STATE['inflight'] -= 1
    call = counted
    try: return await run_level(n, logmon)
    finally: call = inner

if __name__ == '__main__':
    asyncio.run(main())
