#!/usr/bin/env python3
"""compare.py — side-by-side table of several runs (one results/<label>/summary.json per setup).

  python3 bench/compare.py results/official results/optimized [--baseline official] [--md out.md]

Rows are concurrency levels, columns are setups. Ratios are against --baseline (default: the first run).
Any integrity check that is not PASS is shown next to the number, so an unfair run cannot hide in the table.
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='+')
ap.add_argument('--baseline', default=None)
ap.add_argument('--md', default=None)
A = ap.parse_args()
R = {}
for d in A.runs:
    s = json.load(open(os.path.join(d, 'summary.json')))
    R[s['manifest']['label']] = s
labels = list(R)
base = A.baseline or labels[0]
levels = sorted({L['concurrency'] for s in R.values() for L in s['levels']})
def lv(lab, n): return next((L for L in R[lab]['levels'] if L['concurrency'] == n), None)

METRICS = [('Output tokens/s (whole server)', lambda L: L['output_tokens_per_s'], True, '{:,.0f}'),
           ('Per-user output speed p50 (tok/s)', lambda L: L['per_user_tok_s']['p50'], True, '{:,.1f}'),
           ('TTFT p50 (ms)', lambda L: L['ttft_ms']['p50'], False, '{:,.0f}'),
           ('TTFT p99 (ms)', lambda L: L['ttft_ms']['p99'], False, '{:,.0f}'),
           ('TPOT p50 (ms)', lambda L: L['tpot_ms']['p50'], False, '{:,.1f}'),
           ('TPOT p99 (ms)', lambda L: L['tpot_ms']['p99'], False, '{:,.1f}'),
           ('End-to-end p50 (s)', lambda L: L['e2e_ms']['p50'] / 1000, False, '{:,.1f}'),
           ('Completed requests', lambda L: L['completed'], True, '{:,}'),
           ('Prompt tokens served from cache', lambda L: L['cached_frac'], None, '{:.0%}')]

out = []
out.append('| Setup | Corpus sha256 (first 12) | Idle before / after | Fingerprint stable | Single-user tok/s (idle) |')
out.append('|---|---|---|---|---|')
for lab in labels:
    m = R[lab]['manifest']
    out.append(f"| {lab} | `{m['corpus_sha256'][:12]}` | {m['idle_before']['result']} / {m['idle_after']['result']} | {m['fingerprint_match']} | {m['probe_before']['single_user_tok_s']} |")
for name, f, higher, fmt in METRICS:
    out.append(f'\n**{name}**' + ('' if higher is None else (' — higher is better' if higher else ' — lower is better')) + '\n')
    out.append('| Concurrency | ' + ' | '.join(labels) + ' |')
    out.append('|---:|' + '---:|' * len(labels))
    for n in levels:
        cells = []
        for lab in labels:
            L = lv(lab, n)
            if not L: cells.append('—'); continue
            v = f(L); txt = fmt.format(v) if v is not None else '—'
            b = lv(base, n)
            if lab != base and b and v is not None and higher is not None and f(b):
                txt += f' ({v / f(b):.2f}×)'
            bad = [k for k, c in L['integrity'].items() if c.get('result') not in ('PASS',)]
            if bad: txt += ' ⚠ ' + ','.join(f'{k}:{L["integrity"][k]["result"]}' for k in bad)
            cells.append(txt)
        out.append(f'| {n} | ' + ' | '.join(cells) + ' |')
out.append('\n**Integrity checks per level** (exclusive = no other traffic on the server; fixed_len = identical output lengths; canary = answer accuracy under load)\n')
out.append('| Setup | Concurrency | exclusive | max extra requests seen | fixed_len | canary accuracy | prefill |')
out.append('|---|---:|---|---:|---|---|---|')
for lab in labels:
    for L in R[lab]['levels']:
        I = L['integrity']; e = I.get('exclusive', {}); c = I.get('canary', {})
        out.append(f"| {lab} | {L['concurrency']} | {e.get('result', 'not checked')} | {e.get('max_extra_requests', '—')} | {I.get('fixed_len', {}).get('result', '—')} "
                   f"({I.get('fixed_len', {}).get('exact_frac', '—')}) | {c.get('accuracy', '—')} (n={c.get('n', 0)}) | {I.get('prefill', {}).get('result', '—')} |")
text = '\n'.join(out)
print(text)
if A.md: open(A.md, 'w').write(text + '\n')
