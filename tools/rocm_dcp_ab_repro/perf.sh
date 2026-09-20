#!/usr/bin/env bash
# Fixed-shape throughput + acceptance for one arm, run against the live server.
#   usage: pit2_perf.sh <arm> [node] [conc] [isl] [osl]
set -u
ARM="${1:?arm}"; NODE="${2:-pit2-p03-g02}"; CONC="${3:-16}"; ISL="${4:-2048}"; OSL="${5:-256}"

timeout 2400 ssh -o BatchMode=yes -o ConnectTimeout=15 "$NODE" \
  "ARM='$ARM' CONC='$CONC' ISL='$ISL' OSL='$OSL' bash -s" <<'REMOTE'
set -u
RUN=$(readlink -f "/home/$(whoami)/k3ab/runs/current-$ARM")
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://127.0.0.1:8000/health || echo 000)
[ "$code" = "200" ] || { echo "server not healthy ($code)"; exit 3; }

cat > "$RUN/perf.py" <<'PY'
"""Fixed-shape decode throughput and DSpark acceptance for one arm.

Same prompt shape, same concurrency and the same greedy decode for every arm,
so the only thing that moves between runs is the verify path under test.
"""
import json, os, re, time, urllib.request, concurrent.futures as cf

BASE = "http://127.0.0.1:8000/v1/completions"
METRICS = "http://127.0.0.1:8000/metrics"
ARM = os.environ.get("ARM", "?")
CONC = int(os.environ.get("CONC", "16"))
ISL = int(os.environ.get("ISL", "2048"))
OSL = int(os.environ.get("OSL", "256"))

# Deterministic filler so every arm prefills the same number of tokens.
WORD = "the quick brown fox jumps over the lazy dog "
prompt = ("Repeat the following text analysis task. " + WORD * (ISL // 9))[: ISL * 4]

def scrape():
    try:
        with urllib.request.urlopen(METRICS, timeout=10) as r:
            return r.read().decode()
    except Exception:
        return ""

def metric(text, name):
    """Sum one counter across label sets.

    Matches the metric name exactly up to its label brace: the sibling
    ``*_created`` series carries a unix timestamp, and a prefix match would
    quietly add that to the count.
    """
    tot = 0.0
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        head = line.split(" ", 1)[0]
        if head == name or head.startswith(name + "{"):
            try:
                tot += float(line.rsplit(" ", 1)[1])
            except Exception:
                pass
    return tot

def one(_):
    body = json.dumps({"model": "Kimi-K3", "prompt": prompt,
                       "max_tokens": OSL, "min_tokens": OSL,
                       "temperature": 0.0, "ignore_eos": True}).encode()
    req = urllib.request.Request(BASE, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return time.time() - t0, d["usage"]["completion_tokens"]

# Warm the server so compile/JIT does not land inside the measured window.
list(cf.ThreadPoolExecutor(max_workers=CONC).map(one, range(CONC)))

m0 = scrape()
t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=CONC) as ex:
    res = list(ex.map(one, range(CONC * 3)))
wall = time.time() - t0
m1 = scrape()

out_tokens = sum(n for _, n in res)
lat = sorted(t for t, _ in res)
def delta(name):
    return metric(m1, name) - metric(m0, name)

# A "draft" is one speculative step; acceptance length counts the bonus token
# the target always emits on top of whatever drafts survived verification.
drafts = delta("vllm:spec_decode_num_drafts_total")
draft = delta("vllm:spec_decode_num_draft_tokens_total")
acc = delta("vllm:spec_decode_num_accepted_tokens_total")
accept_len = (1 + acc / drafts) if drafts else None

r = {"arm": ARM, "conc": CONC, "isl": ISL, "osl": OSL,
     "requests": len(res), "wall_s": round(wall, 2),
     "output_tok_s": round(out_tokens / wall, 1),
     "lat_p50_s": round(lat[len(lat)//2], 2),
     "lat_p90_s": round(lat[int(len(lat)*0.9)], 2),
     "drafts": drafts, "draft_tokens": draft, "accepted_tokens": acc,
     "accept_len": round(accept_len, 3) if accept_len else None}
print(json.dumps(r))
open("/run_out/perf_result.json", "w").write(json.dumps(r))
PY

docker exec -e ARM="$ARM" -e CONC="$CONC" -e ISL="$ISL" -e OSL="$OSL" \
  "k3ab-$ARM" python3 /run_out/perf.py 2>&1 | tail -6
REMOTE
