#!/usr/bin/env python3
import concurrent.futures
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
GSM8K_TRAIN_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/train.jsonl"
)
DEFAULT_MODEL = (
    "/data/hf-hub-cache/models--amd--Kimi-K2.5-MXFP4/"
    "snapshots/6b0ab7ed538724ea46517351234660bdf36e2d73"
)
MODEL = os.environ.get("MODEL", DEFAULT_MODEL)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def download_if_needed(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    path.write_bytes(data)


def normalize_number(s: str) -> str:
    s = s.strip()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = re.sub(r"[^0-9+\-./]", "", s)
    if s.endswith(".0"):
        s = s[:-2]
    return s


def gold_answer(answer: str) -> str:
    if "####" in answer:
        return normalize_number(answer.split("####")[-1])
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", answer)
    return normalize_number(nums[-1]) if nums else ""


def pred_answer(text: str) -> str:
    if "####" in text:
        tail = text.split("####")[-1]
        nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", tail)
        if nums:
            return normalize_number(nums[0])
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return normalize_number(nums[-1]) if nums else ""


def make_fewshot_prefix(train_items: list[dict], n_shots: int) -> str:
    if n_shots <= 0:
        return ""
    blocks = []
    for i, item in enumerate(train_items[:n_shots], 1):
        blocks.append(
            f"Example {i}:\n"
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}"
        )
    return (
        "Here are solved examples. Follow the format exactly.\n\n"
        + "\n\n".join(blocks)
        + "\n\n"
    )


def make_prompt(question: str, fewshot_prefix: str = "") -> str:
    pad_repeat = env_int("PAD_REPEAT", 0)
    pad = ""
    if pad_repeat > 0:
        pad = (
            "Ignore the following filler words. They are unrelated to the math problem.\n"
            + ("filler " * pad_repeat)
            + "\n\n"
        )
    return (
        fewshot_prefix
        + pad
        + "Solve the following grade-school math problem. "
        "Reason step by step, and end with exactly one line in the form "
        "'#### <number>'.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def post_json(port: int, path: str, payload: dict, timeout: int = 720) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tokenize_count(port: int, prompt: str) -> int:
    data = post_json(port, "/tokenize", {"model": MODEL, "prompt": prompt}, timeout=120)
    if "tokens" in data:
        return len(data["tokens"])
    if "token_ids" in data:
        return len(data["token_ids"])
    if "count" in data:
        return int(data["count"])
    raise RuntimeError(f"unknown tokenize response: {data}")


def align_prompt(port: int, prompt: str, align: int) -> tuple[str, int]:
    if align <= 1:
        return prompt, -1
    for _ in range(align + 4):
        count = tokenize_count(port, prompt)
        if count % align == 0:
            return prompt, count
        prompt += " filler"
    count = tokenize_count(port, prompt)
    return prompt, count


def post_completion(port: int, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    return post_json(port, "/v1/completions", payload, timeout=720)


def get_metrics(port: int) -> str:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
            return r.read().decode()
    except Exception:
        return ""


def parse_metric(text: str, metric: str) -> float:
    vals = []
    prefix = f"vllm:{metric}"
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            vals.append(float(line.rsplit(" ", 1)[-1]))
        except ValueError:
            pass
    return sum(vals)


def main() -> None:
    port = env_int("PORT", 18089)
    n = env_int("N", 64)
    offset = env_int("OFFSET", 0)
    concurrency = env_int("CONCURRENCY", 8)
    max_tokens = env_int("MAX_TOKENS", 256)
    pad_repeat = env_int("PAD_REPEAT", 0)
    align_tokens = env_int("ALIGN_TOKENS", 1)
    label = os.environ.get("LABEL", f"port{port}")
    run = Path(os.environ.get("RUN", "/tmp/kimi_mtp_dcp_verify"))
    data_path = Path(os.environ.get("GSM8K_PATH", "/tmp/kimi_mtp_dcp_verify/data/gsm8k_test.jsonl"))
    train_path = Path(
        os.environ.get("GSM8K_TRAIN_PATH", "/tmp/kimi_mtp_dcp_verify/data/gsm8k_train.jsonl")
    )
    n_shots = env_int("N_SHOTS", 0)
    out_dir = run / "gsm8k_bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    download_if_needed(data_path, GSM8K_TEST_URL)
    train_items = []
    if n_shots > 0:
        download_if_needed(train_path, GSM8K_TRAIN_URL)
        train_items = [
            json.loads(line)
            for line in train_path.read_text().splitlines()
            if line.strip()
        ]
    fewshot_prefix = make_fewshot_prefix(train_items, n_shots)
    items = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    sample = items[offset : offset + n]
    if len(sample) != n:
        raise SystemExit(f"requested {n} samples from offset {offset}, got {len(sample)}")

    metrics_before = get_metrics(port)
    before = {
        "request_success_total": parse_metric(metrics_before, "request_success_total"),
        "prompt_tokens_total": parse_metric(metrics_before, "prompt_tokens_total"),
        "generation_tokens_total": parse_metric(metrics_before, "generation_tokens_total"),
        "spec_decode_num_drafts_total": parse_metric(metrics_before, "spec_decode_num_drafts_total"),
        "spec_decode_num_draft_tokens_total": parse_metric(metrics_before, "spec_decode_num_draft_tokens_total"),
        "spec_decode_num_accepted_tokens_total": parse_metric(metrics_before, "spec_decode_num_accepted_tokens_total"),
    }

    prepared = []
    prep_started = time.perf_counter()
    for idx, item in enumerate(sample):
        prompt, prompt_token_count = align_prompt(port, make_prompt(item["question"], fewshot_prefix), align_tokens)
        prepared.append((idx, item, prompt, prompt_token_count))
    prep_wall = time.perf_counter() - prep_started

    results = []
    started = time.perf_counter()

    def run_one(pair: tuple[int, dict, str, int]) -> dict:
        idx, item, prompt, prompt_token_count = pair
        t0 = time.perf_counter()
        try:
            resp = post_completion(port, prompt, max_tokens)
            latency = time.perf_counter() - t0
            choice = resp["choices"][0]
            text = choice.get("text", "")
            usage = resp.get("usage", {})
            gold = gold_answer(item["answer"])
            pred = pred_answer(text)
            return {
                "idx": offset + idx,
                "ok": True,
                "latency_s": latency,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "prompt_token_count_pre": prompt_token_count,
                "gold": gold,
                "pred": pred,
                "exact": pred == gold,
                "finish_reason": choice.get("finish_reason"),
                "question": item["question"],
                "output": text,
            }
        except Exception as e:
            return {
                "idx": offset + idx,
                "ok": False,
                "latency_s": time.perf_counter() - t0,
                "error": repr(e),
                "question": item["question"],
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(run_one, pair) for pair in prepared]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    wall = time.perf_counter() - started
    results.sort(key=lambda x: x["idx"])
    metrics_after = get_metrics(port)
    after = {
        "request_success_total": parse_metric(metrics_after, "request_success_total"),
        "prompt_tokens_total": parse_metric(metrics_after, "prompt_tokens_total"),
        "generation_tokens_total": parse_metric(metrics_after, "generation_tokens_total"),
        "spec_decode_num_drafts_total": parse_metric(metrics_after, "spec_decode_num_drafts_total"),
        "spec_decode_num_draft_tokens_total": parse_metric(metrics_after, "spec_decode_num_draft_tokens_total"),
        "spec_decode_num_accepted_tokens_total": parse_metric(metrics_after, "spec_decode_num_accepted_tokens_total"),
    }
    delta = {k: after[k] - before[k] for k in before}

    oks = [r for r in results if r.get("ok")]
    exact = [r for r in oks if r.get("exact")]
    completion_tokens = sum(int(r.get("completion_tokens", 0)) for r in oks)
    prompt_tokens = sum(int(r.get("prompt_tokens", 0)) for r in oks)
    latencies = [float(r["latency_s"]) for r in oks]
    summary = {
        "label": label,
        "port": port,
        "n": n,
        "offset": offset,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "pad_repeat": pad_repeat,
        "n_shots": n_shots,
        "align_tokens": align_tokens,
        "prep_wall_s": prep_wall,
        "ok": len(oks),
        "errors": n - len(oks),
        "exact": len(exact),
        "accuracy": (len(exact) / len(oks)) if oks else 0.0,
        "wall_s": wall,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "completion_tok_s_client": completion_tokens / wall if wall else 0.0,
        "total_tok_s_client": (prompt_tokens + completion_tokens) / wall if wall else 0.0,
        "req_s_client": len(oks) / wall if wall else 0.0,
        "latency_avg_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_s": statistics.median(latencies) if latencies else 0.0,
        "latency_max_s": max(latencies) if latencies else 0.0,
        "metrics_delta": delta,
    }
    if delta["spec_decode_num_draft_tokens_total"] > 0:
        summary["server_spec_accept_rate"] = (
            delta["spec_decode_num_accepted_tokens_total"]
            / delta["spec_decode_num_draft_tokens_total"]
        )

    result_path = out_dir / f"{label}_results.jsonl"
    summary_path = out_dir / f"{label}_summary.json"
    result_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"RESULTS {result_path}")
    print(f"SUMMARY {summary_path}")


if __name__ == "__main__":
    main()
