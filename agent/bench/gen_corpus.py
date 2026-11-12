#!/usr/bin/env python3
"""gen_corpus.py — build a deterministic coding-agent request corpus.

Each *session* is one agent working on one task in a repository: a shared agent system prompt + tool schemas,
a user task, then alternating assistant tool calls and tool results (file reads, grep hits, test output, diffs).
One line per session: the full message list plus, for each request, how many leading messages it sends
(`n_messages`) and its output length. Request k = the conversation up to step k, so consecutive requests share a growing prefix, exactly
like a real coding agent. Code text is taken from a Python standard-library tree (real, varied source code).

The corpus that ships in corpus/ was generated with:  python3 bench/gen_corpus.py --seed 20261111 --out corpus/coding_v1.jsonl.gz
(Python 3.10 stdlib). Regenerating on another Python version gives different text, so compare setups with the
shipped file; every run records the corpus sha256.
"""
import argparse, ast, gzip, json, os, random, sysconfig

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=20261111)
ap.add_argument('--sessions', type=int, default=120)
ap.add_argument('--code-root', default=sysconfig.get_paths()['stdlib'])
ap.add_argument('--chars-per-token', type=float, default=3.3, help='used only to hit length targets; measured on GLM tokenizer')
A = ap.parse_args()
R = random.Random(A.seed)
CPT = A.chars_per_token

SYSTEM = """You are a senior software engineer working as an autonomous coding agent inside a developer's repository. You can read files, search the code base, edit files, and run commands through the tools provided. Your goal is to complete the user's task end to end: understand the relevant code, make a focused change, verify it, and report what you did.

# How to work
- Start by locating the code that matters. Prefer searching (grep_code) and listing directories over guessing paths. Read the surrounding code before changing it, including callers and tests.
- Make the smallest change that fully solves the task. Do not reformat unrelated code, rename public APIs, or add dependencies unless the task requires it.
- Follow the conventions already used in the file: naming, typing style, docstring format, error handling, logging, and import order.
- After every edit, run the most relevant tests. If a test fails, read the failure carefully, decide whether the test or the code is wrong, and fix the right one. Never delete or weaken a test just to make it pass.
- If the task is ambiguous, choose the interpretation most consistent with the existing code and state your assumption in the final summary.
- Keep the user informed with short progress notes between tool calls: what you found and what you will do next. Do not narrate every keystroke.

# Editing rules
- Use edit_file with an exact `old` snippet that appears once in the file; include enough context lines to make it unique.
- Preserve indentation exactly. Python code uses four spaces.
- When adding a function, add or update its docstring and type hints, and add a unit test next to the existing tests for that module.
- When fixing a bug, first write or identify a test that reproduces it, then fix the code, then show the test passing.
- Do not leave debugging prints, commented-out code, or TODOs that you introduced.

# Running commands
- Use run_command for tests, linters and short scripts. Prefer the narrowest test selection (a single test file or test id) while iterating, then run the whole module's tests once at the end.
- Commands run in the repository root with a 300 second timeout. Long-running servers or interactive programs are not allowed.
- Treat command output as data. If output is truncated, narrow the command rather than re-running the same thing.

# Safety
- Never print, copy, or commit secrets, tokens, credentials or private keys, even if they appear in files you read.
- Do not run destructive commands (rm -rf, git reset --hard, force pushes, dropping databases) unless the user explicitly asked for exactly that.
- Do not access the network except through the package manager when the task requires installing a dependency that the project already declares.

# Final answer
When the task is done, reply with: a one-paragraph summary of the change, the list of files touched, the tests you ran and their result, and any follow-up the user should consider. Keep it concise and factual; do not claim tests passed unless you ran them in this session.
"""

def tool(name, desc, props, req):
    return {'type': 'function', 'function': {'name': name, 'description': desc,
            'parameters': {'type': 'object', 'properties': props, 'required': req}}}
S = lambda d: {'type': 'string', 'description': d}
I = lambda d: {'type': 'integer', 'description': d}
TOOLS = [
    tool('read_file', 'Read a text file from the repository. Returns the requested line range with line numbers. Use start/end to page through large files instead of reading everything at once.',
         {'path': S('Path relative to the repository root.'), 'start': I('First line to return, 1-based. Default 1.'), 'end': I('Last line to return, inclusive. Default: start + 400.')}, ['path']),
    tool('grep_code', 'Search the repository with a regular expression. Returns matching lines as path:line: text, at most 200 matches. Use include to restrict to a glob such as "*.py".',
         {'pattern': S('Python regular expression.'), 'include': S('Optional glob filter, e.g. "*.py" or "tests/**".'), 'ignore_case': {'type': 'boolean', 'description': 'Case-insensitive match.'}}, ['pattern']),
    tool('list_dir', 'List files and directories under a path, up to the given depth. Hidden files and build artifacts are skipped.',
         {'path': S('Directory relative to the repository root.'), 'depth': I('How many levels to descend. Default 2.')}, ['path']),
    tool('edit_file', 'Replace exactly one occurrence of `old` with `new` in a file. Fails if `old` is not found or is found more than once. To create a new file, pass an empty `old` and the full content as `new`.',
         {'path': S('File to edit.'), 'old': S('Exact text to replace, including indentation.'), 'new': S('Replacement text.')}, ['path', 'old', 'new']),
    tool('run_command', 'Run a shell command in the repository root and return exit code, stdout and stderr (truncated to the last 400 lines). Use for tests, linters, type checkers and short scripts.',
         {'command': S('The command line to execute.'), 'timeout_s': I('Timeout in seconds, at most 300.')}, ['command']),
    tool('git_diff', 'Show the unified diff of uncommitted changes, optionally limited to one path.', {'path': S('Optional path filter.')}, []),
]

# ---------- code pool ----------
pool = []
for root, dirs, files in os.walk(A.code_root):
    dirs[:] = sorted(d for d in dirs if d not in ('site-packages', 'dist-packages', '__pycache__', 'idlelib', 'turtledemo', 'lib2to3', 'ensurepip'))
    for f in sorted(files):
        if f.endswith('.py'):
            p = os.path.join(root, f)
            try:
                src = open(p, encoding='utf-8').read()
                tree = ast.parse(src)
            except Exception:
                continue
            if not 3000 < len(src) < 300000: continue
            funcs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith('__')]
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            if len(funcs) < 3: continue
            rel = os.path.relpath(p, A.code_root)
            pool.append(dict(path='src/' + rel.replace('test/', 'tests/'), lines=src.splitlines(), funcs=funcs, classes=classes))
pool.sort(key=lambda x: x['path'])
assert len(pool) > 50, f'too few source files under {A.code_root}'

def numbered(lines, start):
    return '\n'.join(f'{start + i:>5}\t{l}' for i, l in enumerate(lines))

ERRS = ['TypeError', 'ValueError', 'KeyError', 'AttributeError', 'IndexError', 'AssertionError', 'RuntimeError', 'UnicodeDecodeError']
TASKS = ['Fix the failing test test_{f} in tests/test_{m}.py. It started failing after the last refactor of {f}().',
         '{f}() raises {e} when the input is empty. Make it return a sensible default instead and add a regression test.',
         'Add type hints to the public functions in {p}, starting with {f}(). Keep runtime behaviour identical.',
         'Refactor {c} so that the parsing logic in {f}() is a separate private helper. No behaviour change; tests must still pass.',
         'Our profiler shows {f}() in {p} is a hotspot. Find an obvious inefficiency and fix it without changing results.',
         'Users report that {f}() silently ignores invalid arguments. Validate the arguments and raise a clear {e}. Update the docstring.',
         'Write unit tests for {f}() covering the edge cases you can find in its implementation.',
         'Explain how {c} and {f}() interact, then fix the off-by-one error reported in issue #{n} (the last element is dropped).']

def task_for(src):
    f = R.choice(src['funcs']); c = R.choice(src['classes']) if src['classes'] else f
    m = os.path.splitext(os.path.basename(src['path']))[0]
    return R.choice(TASKS).format(f=f, m=m, p=src['path'], c=c, e=R.choice(ERRS), n=R.randint(120, 9800)), f

def step_read(src, f):
    n = len(src['lines']); span = R.randint(80, 360)
    idx = next((i for i, l in enumerate(src['lines']) if f'def {f}(' in l), R.randrange(n))
    a = max(0, idx - R.randint(0, span // 2)); b = min(n, a + span)
    return ('read_file', {'path': src['path'], 'start': a + 1, 'end': b}, numbered(src['lines'][a:b], a + 1),
            R.choice(['Let me read the implementation first.', f'Reading {src["path"]} around {f}().', 'I need the surrounding code before editing.']))

def step_grep(src, f):
    hits = []
    for s in R.sample(pool, 12):
        for i, l in enumerate(s['lines']):
            if f in l or (R.random() < 0.004 and l.strip()):
                hits.append(f'{s["path"]}:{i + 1}: {l.strip()[:160]}')
    hits = hits[:R.randint(20, 120)] or [f'{src["path"]}:1: # no other references']
    return ('grep_code', {'pattern': rf'\b{f}\b', 'include': '*.py'}, '\n'.join(hits), f'Searching for other uses of {f}.')

def step_tests(src, f):
    s2 = R.choice(pool); frames = []
    for _ in range(R.randint(3, 8)):
        s = R.choice(pool); i = R.randrange(len(s['lines']))
        frames.append(f'  File "{s["path"]}", line {i + 1}, in {R.choice(s["funcs"])}\n    {s["lines"][i].strip()[:120]}')
    e = R.choice(ERRS); npass = R.randint(20, 400); nfail = R.randint(1, 4)
    body = (f'============================= test session starts ==============================\nplatform linux -- Python 3.11.9, pytest-8.3.2\n'
            f'collected {npass + nfail} items\n\n' + '\n'.join(f'tests/test_{os.path.basename(s2["path"])[:-3]}.py::test_{g} PASSED' for g in R.sample(s2['funcs'], min(len(s2['funcs']), R.randint(8, 40)))) +
            f'\n\n=================================== FAILURES ===================================\n___________________________ test_{f} ___________________________\nTraceback (most recent call last):\n' +
            '\n'.join(frames) + f'\n{e}: unexpected value in {f}()\n========================= {nfail} failed, {npass} passed in {R.uniform(0.8, 40):.2f}s =========================')
    return ('run_command', {'command': f'python -m pytest tests/ -x -q -k {f}', 'timeout_s': 300}, f'exit code: 1\n{body}', 'Running the relevant tests.')

def step_diff(src, f):
    n = len(src['lines']); a = R.randrange(max(1, n - 60)); old = src['lines'][a:a + R.randint(8, 30)]
    new = [l.replace('return ', 'return _checked(') + (')' if 'return ' in l else '') for l in old]
    d = [f'diff --git a/{src["path"]} b/{src["path"]}', f'--- a/{src["path"]}', f'+++ b/{src["path"]}', f'@@ -{a + 1},{len(old)} +{a + 1},{len(new)} @@']
    d += [f'-{l}' for l in old] + [f'+{l}' for l in new]
    return ('git_diff', {'path': src['path']}, '\n'.join(d), 'Checking the diff so far.')

def step_ls(src, f):
    base = os.path.dirname(src['path']); items = sorted({p['path'] for p in pool if p['path'].startswith(base)})[:R.randint(15, 80)]
    return ('list_dir', {'path': base, 'depth': 2}, '\n'.join(items) or base, 'Listing the package layout.')

STEPS = [(step_read, 5), (step_grep, 2), (step_tests, 2), (step_diff, 1), (step_ls, 1)]
def pick_step():
    t = R.uniform(0, sum(w for _, w in STEPS))
    for s, w in STEPS:
        t -= w
        if t <= 0: return s
    return STEPS[0][0]

def out_len(final):
    if final: return R.randint(400, 1200)
    return R.randint(64, 256) if R.random() < 0.65 else R.randint(256, 800)

rows = []; sessions = []
base_tokens = (len(SYSTEM) + len(json.dumps(TOOLS))) / CPT
for s_i in range(A.sessions):
    src = R.choice(pool); task, f = task_for(src)
    # session context cap: 40% 12-24K, 40% 24-40K, 20% 40-64K tokens
    u = R.random(); cap = R.randint(12000, 24000) if u < .4 else R.randint(24000, 40000) if u < .8 else R.randint(40000, 64000)
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task}]
    steps = [step_read]
    k = 0
    while True:
        st = steps.pop(0) if steps else pick_step()
        s_src = src if R.random() < 0.6 else R.choice(pool)
        name, args, result, note = st(s_src, f if st is not step_read or s_src is src else R.choice(s_src['funcs']))
        cid = f'call_{s_i:03d}_{k:02d}'
        msgs.append({'role': 'assistant', 'content': note, 'tool_calls': [{'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]})
        msgs.append({'role': 'tool', 'tool_call_id': cid, 'content': result})
        est = base_tokens + sum(len(m['content']) for m in msgs[1:]) / CPT
        final = est > cap
        rows.append({'id': f's{s_i:03d}-t{k:02d}', 'n_messages': len(msgs), 'est_input_tokens': int(est), 'max_tokens': out_len(final)})
        k += 1
        if final: break
    sessions.append({'session': f's{s_i:03d}', 'messages': msgs, 'requests': rows[-k:]})

os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
with gzip.open(A.out, 'wt', encoding='utf-8') as fo:
    fo.write(json.dumps({'_meta': {'generator': 'gen_corpus.py', 'seed': A.seed, 'sessions': A.sessions, 'tools': TOOLS}}) + '\n')
    for x in sessions: fo.write(json.dumps(x, ensure_ascii=False) + '\n')
ins = sorted(r['est_input_tokens'] for r in rows); outs = sorted(r['max_tokens'] for r in rows)
q = lambda xs, p: xs[int(p * (len(xs) - 1))]
print(f'{len(rows)} requests in {A.sessions} sessions; est input tokens p10/p50/p90/max = {q(ins,.1)}/{q(ins,.5)}/{q(ins,.9)}/{ins[-1]}; '
      f'output p10/p50/p90 = {q(outs,.1)}/{q(outs,.5)}/{q(outs,.9)}')
