"""provenance.py — who ran what, when, against what, and whether anyone else was using the server.

Two things this answers for every run:
  1. Provenance: UTC start/end, hashes of the corpus / script / output, versions, arguments, server identity.
  2. Exclusivity: was the server serving only us?  Checked three ways, all read from the server's Prometheus
     /metrics endpoint when it has one (SGLang, vLLM and compatible engines do):
       - idle check before we start (running/queued requests must be 0),
       - a sampler during the run: more running requests than our concurrency means somebody else is sending traffic,
       - token accounting: tokens the server says it generated over the whole run vs. tokens we received.
     Engines without /metrics get status "unverified": the result is then only as exclusive as the tester says it is.
"""
import hashlib, json, os, platform, re, socket, subprocess, sys

RUNNING = ('num_running_reqs', 'num_requests_running')
QUEUED = ('num_queue_reqs', 'num_requests_waiting')
GEN = ('generation_tokens_total',)
LINE = re.compile(r'^(?:[A-Za-z_]+:)?([A-Za-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$')

def sha256_file(p, n=None):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def parse_metrics(text):
    """-> {'running': x, 'queued': x, 'gen_tokens': x} (None for what the engine does not expose)"""
    acc = {}
    for ln in text.splitlines():
        m = LINE.match(ln)
        if m: acc[m.group(1)] = acc.get(m.group(1), 0.0) + float(m.group(2))
    pick = lambda names: next((acc[n] for n in names if n in acc), None)
    return dict(running=pick(RUNNING), queued=pick(QUEUED), gen_tokens=pick(GEN))

async def scrape(sess, url):
    try:
        async with sess.get(url + '/metrics') as r:
            if r.status == 200: return parse_metrics(await r.text())
    except Exception: pass
    return None

async def server_identity(sess, url, out_dir):
    """Store what the server says about itself next to the results (it can hold engine settings: the tester decides whether to publish it)."""
    ident = {}
    for path in ('/v1/models', '/get_server_info', '/server_info', '/version'):
        try:
            async with sess.get(url + path) as r:
                if r.status == 200:
                    body = await r.read(); ident[path] = hashlib.sha256(body).hexdigest()[:16]
                    open(f'{out_dir}/server{path.replace("/", "_")}.json', 'wb').write(body)
        except Exception: pass
    return ident

def environment(corpus, script_dir):
    git = None
    try: git = subprocess.run(['git', '-C', script_dir, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception: pass
    return dict(client_host_hash=hashlib.sha256(socket.gethostname().encode()).hexdigest()[:8], python=platform.python_version(),
                platform=platform.platform(), aiohttp=sys.modules['aiohttp'].__version__,
                corpus_sha256=sha256_file(corpus), replay_py_sha256=sha256_file(os.path.join(script_dir, 'replay.py')), repo_commit=git)

def verdict(before, after, samples, concurrency, client_out):
    """Decide clean / foreign_traffic_suspected / unverified from the scrapes."""
    if before is None or after is None or before['gen_tokens'] is None:
        return dict(status='unverified', reason='server exposes no /metrics; exclusivity is the tester\'s claim')
    why = []
    peak = max((x for x in samples if x is not None), default=None)
    if peak is not None and peak > concurrency: why.append(f'saw {peak:.0f} running requests, we only send {concurrency}')
    srv = after['gen_tokens'] - before['gen_tokens']
    ratio = srv / client_out if client_out else None
    if ratio is not None and ratio > 1.05: why.append(f'server generated {srv:.0f} tokens but we received {client_out} (x{ratio:.2f})')
    return dict(status='foreign_traffic_suspected' if why else 'clean', reason='; '.join(why) or 'running-request peak and token accounting match our own traffic',
                running_peak=peak, server_generated_tokens=srv, client_received_tokens=client_out, server_to_client_ratio=None if ratio is None else round(ratio, 3))
