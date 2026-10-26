#!/usr/bin/env python3
"""End-to-end smoke test: local Chroma fixtures -> real Pinecone indexes.

Needs PINECONE_API_KEY in the environment. Creates indexes named
chroma-mig-eng-<run>-{l2,cos,ip,fts} and deletes them at the end, even on
failure. Every step has an expected exit code; the run fails if any differs.

  python tests/smoke_test.py --scratch /path/to/empty/dir
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
PREFIX = "chroma-mig-eng-"


class Runner:
    def __init__(self, scratch: str):
        self.scratch = scratch
        self.chroma = os.path.join(scratch, "chroma")
        self.results: list[dict] = []

    def wd(self, name: str) -> str:
        return os.path.join(self.scratch, name)

    def run(self, label: str, script: str, args: list[str], expect: int, env: dict | None = None) -> str:
        cmd = [PY, os.path.join(ROOT, script), *args]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
        ok = p.returncode == expect
        tail = (p.stdout + p.stderr).strip().splitlines()[-6:]
        self.results.append({"step": label, "script": script, "exit": p.returncode, "expected": expect,
                             "ok": ok, "seconds": round(time.time() - t0, 1), "tail": tail})
        print(f"{'PASS' if ok else 'FAIL'} [{p.returncode}/{expect}] {label} ({time.time() - t0:.1f}s)")
        if not ok or os.environ.get("SMOKE_VERBOSE"):
            print("\n".join("    " + line[:300] for line in (p.stdout + p.stderr).splitlines()[-40:]))
        return p.stdout

    def note(self, label: str, ok: bool, detail: str = ""):
        self.results.append({"step": label, "ok": ok, "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} {label} {detail}")


def chroma(r: Runner, collection: str) -> list[str]:
    return ["--collection", collection, "--chroma-path", r.chroma]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--keep-indexes", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("PINECONE_API_KEY"):
        print("PINECONE_API_KEY is not set")
        return 2
    shutil.rmtree(args.scratch, ignore_errors=True)
    os.makedirs(args.scratch)
    r = Runner(args.scratch)
    run_id = uuid.uuid4().hex[:6]
    idx = {k: f"{PREFIX}{run_id}-{k}" for k in ("l2", "cos", "ip", "fts")}
    assert all(n.startswith(PREFIX) and len(n) <= 45 for n in idx.values())
    filters = os.path.join(HERE, "filters_qa_main.json")
    golden = ["--golden-from-sample", "30"]

    sys.path.insert(0, ROOT)
    from pinecone import Pinecone

    pc = Pinecone()
    try:
        r.run("unit checks", "tests/test_convert_unit.py", [], 0)
        r.run("build fixtures", "tests/build_fixtures.py", ["--chroma-path", r.chroma], 0)

        w = r.wd("w_main")
        main_c = chroma(r, "qa_main")
        r.run("main: preflight, default policies (arrays block)", "preflight.py", main_c + ["--workdir", w], 1)
        r.run("main: preflight, stringify, transforms not yet allowed", "preflight.py",
              main_c + ["--workdir", w, "--array-policy", "stringify"], 1)
        r.run("main: preflight, stringify + --allow-transforms + filters", "preflight.py",
              main_c + ["--workdir", w, "--array-policy", "stringify", "--allow-transforms", "--app-filters", filters], 0)
        r.run("main: create euclidean index", "create_index.py", ["--index", idx["l2"], "--workdir", w], 0)
        r.run("main: create_index rerun verifies schema", "create_index.py", ["--index", idx["l2"], "--workdir", w], 0)

        import chromadb

        col = chromadb.PersistentClient(path=r.chroma).get_collection("qa_main", embedding_function=None)
        col.update(ids=["doc-0005"], embeddings=[[0.3, -1.2, 0.8, 2.0, -0.4, 1.1, 0.0, 0.7]],
                   documents=["updated before the pause"],
                   metadatas=[{"category": "updated", "year": 2024}])
        col.delete(ids=["doc-0006"])
        r.note("main: update + delete before the pause", col.count() == 250, f"count={col.count()}")

        r.run("main: export, page 50 (exact multiple of 250)", "export.py", main_c + ["--workdir", w, "--page-size", "50"], 0)
        w60 = r.wd("w_main_p60")
        r.run("main: export, page 60 (partial last page)", "export.py", main_c + ["--workdir", w60, "--page-size", "60"], 0)
        m50 = json.load(open(os.path.join(w, "export_manifest.json")))
        m60 = json.load(open(os.path.join(w60, "export_manifest.json")))
        r.note("main: both page sizes export the same 250 unique IDs and bytes",
               m50["unique_ids"] == m60["unique_ids"] == 250 and m50["sha256"] == m60["sha256"],
               f"{m50['unique_ids']}/{m60['unique_ids']}")
        r.run("main: convert, default policy rejects arrays", "convert.py", ["--workdir", w], 1)
        r.note("main: rejected convert leaves no documents.jsonl", not os.path.exists(os.path.join(w, "documents.jsonl")))
        r.note("main: failed convert left transforms.jsonl.failed", os.path.exists(os.path.join(w, "transforms.jsonl.failed")))
        r.run("main: convert --array-policy stringify", "convert.py", ["--workdir", w, "--array-policy", "stringify"], 0)
        r.note("main: successful convert removed the stale transforms.jsonl.failed",
               not os.path.exists(os.path.join(w, "transforms.jsonl.failed")))
        r.run("main: load", "load.py", ["--index", idx["l2"], "--namespace", "qa_main", "--workdir", w], 0)
        st = json.load(open(os.path.join(w, "load_state.json")))
        r.note("main: size-aware batching split the load", st["batches"] > 1 and st["max_batch_bytes"] <= 1_900_000,
               f"batches={st['batches']} max_batch_bytes={st['max_batch_bytes']}")
        r.run("main: validate (+ filter parity)", "validate.py",
              main_c + ["--index", idx["l2"], "--namespace", "qa_main", "--workdir", w, "--app-filters", filters,
                        "--report", os.path.join(w, "validation_report_first.json")] + golden, 0)
        first = json.load(open(os.path.join(w, "validation_report_first.json")))
        frows = {f["name"]: f for c in first["checks"] if c["check"] == "app_filters" for f in c["filters"]}
        r.note("main: $contains on a scalar key is flagged, not rewritten",
               frows["category_contains_scalar"]["status"] == "needs_redesign"
               and frows["category_contains_scalar"]["pinecone_filter"] is None)
        r.note("main: $contains on an array key is still rewritten and matches",
               frows["tags_contains"]["same_ids"] is True)

        missing = f"{PREFIX}{run_id}-missing"
        r.run("S2: validate with a misspelled collection exits 2", "validate.py",
              ["--collection", "qa_mian", "--chroma-path", r.chroma, "--index", idx["l2"], "--namespace", "qa_main",
               "--workdir", w] + golden, 2)
        r.run("S2: validate with a missing index exits 2", "validate.py",
              main_c + ["--index", missing, "--namespace", "qa_main", "--workdir", w] + golden, 2)
        r.run("S2: load with a missing index exits 2", "load.py",
              ["--index", missing, "--namespace", "qa_main", "--workdir", w], 2)
        nokey = {k: v for k, v in os.environ.items() if k != "PINECONE_API_KEY"}
        r.run("S2: create_index without PINECONE_API_KEY exits 2", "create_index.py",
              ["--index", idx["l2"], "--workdir", w], 2, env=nokey)
        r.run("S2: load without PINECONE_API_KEY exits 2", "load.py",
              ["--index", idx["l2"], "--namespace", "qa_main", "--workdir", w], 2, env=nokey)
        r.run("S2: validate without PINECONE_API_KEY exits 2", "validate.py",
              main_c + ["--index", idx["l2"], "--namespace", "qa_main", "--workdir", w] + golden, 2, env=nokey)
        r.run("main: loader rerun", "load.py", ["--index", idx["l2"], "--namespace", "qa_main", "--workdir", w], 0)
        r.run("main: validate after rerun (no-op)", "validate.py",
              main_c + ["--index", idx["l2"], "--namespace", "qa_main", "--workdir", w] + golden, 0)
        wcopy = r.wd("w_main_copy")
        shutil.copytree(w, wcopy)
        os.remove(os.path.join(wcopy, "load_state.json"))
        r.run("main: load refuses a non-empty namespace without matching state", "load.py",
              ["--index", idx["l2"], "--namespace", "qa_main", "--workdir", wcopy], 2)

        for name, space_idx, ns in (("qa_cosine", "cos", "qa_cosine"), ("qa_ip", "ip", "qa_ip")):
            wd = r.wd("w_" + name)
            c = chroma(r, name)
            r.run(f"{name}: preflight", "preflight.py", c + ["--workdir", wd], 0)
            r.run(f"{name}: create index", "create_index.py", ["--index", idx[space_idx], "--workdir", wd], 0)
            r.run(f"{name}: export", "export.py", c + ["--workdir", wd], 0)
            r.run(f"{name}: convert", "convert.py", ["--workdir", wd], 0)
            r.run(f"{name}: load", "load.py", ["--index", idx[space_idx], "--namespace", ns, "--workdir", wd], 0)
            r.run(f"{name}: validate", "validate.py", c + ["--index", idx[space_idx], "--namespace", ns,
                                                            "--workdir", wd] + golden, 0)

        we = r.wd("w_empty")
        ce = chroma(r, "qa_empty")
        r.run("qa_empty: preflight (warning only)", "preflight.py", ce + ["--workdir", we], 0)
        r.run("qa_empty: export", "export.py", ce + ["--workdir", we], 0)
        r.run("qa_empty: convert", "convert.py", ["--workdir", we], 0)
        r.run("qa_empty: load (0 documents)", "load.py", ["--index", idx["l2"], "--namespace", "qa_empty", "--workdir", we], 0)
        r.run("qa_empty: validate", "validate.py", ce + ["--index", idx["l2"], "--namespace", "qa_empty", "--workdir", we], 0)
        rep_e = json.load(open(os.path.join(we, "validation_report.json")))
        md = next(c for c in rep_e["checks"] if c["check"] == "metric_and_dimension")
        r.note("M8: empty collection reports the dimension as unchecked (warn)",
               md["status"] == "warn" and "not checked" in md["summary"], md["summary"])
        r.run("M8: empty collection, --dimension 8 matches the index", "validate.py",
              ce + ["--index", idx["l2"], "--namespace", "qa_empty", "--workdir", we, "--dimension", "8"], 0)
        r.run("M8: empty collection, --dimension 16 fails", "validate.py",
              ce + ["--index", idx["l2"], "--namespace", "qa_empty", "--workdir", we, "--dimension", "16"], 1)

        wb = r.wd("w_blocked")
        cb = chroma(r, "qa_blocked")
        out = r.run("qa_blocked: preflight blocks", "preflight.py", cb + ["--workdir", wb, "--allow-transforms"], 1)
        r.note("qa_blocked: zero vector and >40 KB metadata both reported",
               "zero_vector" in out and "metadata_over_40kb" in out)
        r.run("qa_blocked: export", "export.py", cb + ["--workdir", wb], 0)
        r.run("qa_blocked: convert refuses", "convert.py", ["--workdir", wb], 1)

        wf = r.wd("w_fts")
        cf = chroma(r, "qa_fts")
        r.run("qa_fts: preflight, metadata mode blocks the 60 KB document", "preflight.py", cf + ["--workdir", wf], 1)
        r.run("qa_fts: preflight --text-mode fts", "preflight.py", cf + ["--workdir", wf, "--text-mode", "fts"], 0)
        r.run("qa_fts: create index --full-text-search", "create_index.py",
              ["--index", idx["fts"], "--workdir", wf, "--full-text-search"], 0)
        r.run("qa_fts: export", "export.py", cf + ["--workdir", wf], 0)
        r.run("qa_fts: convert --text-mode fts", "convert.py", ["--workdir", wf, "--text-mode", "fts"], 0)
        r.run("qa_fts: load", "load.py", ["--index", idx["fts"], "--namespace", "qa_fts", "--workdir", wf], 0)
        r.run("qa_fts: validate", "validate.py", cf + ["--index", idx["fts"], "--namespace", "qa_fts",
                                                       "--workdir", wf] + golden, 0)

        wt = r.wd("w_tok_over")
        ct = chroma(r, "qa_tok_over")
        out = r.run("S1: preflight --text-mode fts blocks 30,000 tokens and a 1,025-byte token", "preflight.py",
                    ct + ["--workdir", wt, "--text-mode", "fts"], 1)
        r.note("S1: both token findings are BLOCKING",
               "[BLOCKING] text_over_10000_tokens" in out and "[BLOCKING] text_token_over_1024_bytes" in out)
        r.run("S1: export qa_tok_over", "export.py", ct + ["--workdir", wt], 0)
        out = r.run("S1: convert --text-mode fts refuses", "convert.py", ["--workdir", wt, "--text-mode", "fts"], 1)
        r.note("S1: convert names both errors",
               "document_text_over_10000_tokens" in out and "document_token_over_1024_bytes" in out)
        wte = r.wd("w_tok_edge")
        cte = chroma(r, "qa_tok_edge")
        r.run("S1: qa_tok_edge preflight fts (10,000 tokens, 1,024-byte token)", "preflight.py",
              cte + ["--workdir", wte, "--text-mode", "fts"], 0)
        r.run("S1: qa_tok_edge export", "export.py", cte + ["--workdir", wte], 0)
        r.run("S1: qa_tok_edge convert", "convert.py", ["--workdir", wte, "--text-mode", "fts"], 0)
        r.run("S1: qa_tok_edge load (server accepts the boundary values)", "load.py",
              ["--index", idx["fts"], "--namespace", "qa_tok_edge", "--workdir", wte], 0)
        r.run("S1: qa_tok_edge validate", "validate.py",
              cte + ["--index", idx["fts"], "--namespace", "qa_tok_edge", "--workdir", wte,
                     "--golden-from-sample", "3"], 0)
        fidx = pc.index(name=idx["fts"])
        for label, text in (("10,001 tokens", "a " * 10001), ("1,025-byte token", "b" * 1025)):
            try:
                fidx.documents.upsert(namespace="probe", documents=[{"_id": "p", "embedding": [1.0] * 8, "text": text}])
                r.note(f"S1: server rejects {label}", False, "accepted")
            except Exception as exc:
                r.note(f"S1: server rejects {label}", "400" in str(exc), str(exc)[:120])

        wm = r.wd("w_mismatch")
        shutil.copytree(w, wm)
        os.remove(os.path.join(wm, "load_state.json"))
        r.run("mismatch: load l2 data into a cosine index is refused", "load.py",
              ["--index", idx["cos"], "--namespace", "mismatch", "--workdir", wm], 1)
        r.run("mismatch: forced load", "load.py",
              ["--index", idx["cos"], "--namespace", "mismatch", "--workdir", wm, "--allow-metric-mismatch"], 0)
        r.run("mismatch: validator fails", "validate.py",
              main_c + ["--index", idx["cos"], "--namespace", "mismatch", "--workdir", wm] + golden, 1)

        ccol = chromadb.PersistentClient(path=r.chroma).get_collection("qa_cosine", embedding_function=None)
        ccol.add(ids=["late-write"], embeddings=[[0.5] * 8], documents=["written after export"])
        r.run("writes not paused: validator fails on a late write", "validate.py",
              chroma(r, "qa_cosine") + ["--index", idx["cos"], "--namespace", "qa_cosine",
                                        "--workdir", r.wd("w_qa_cosine")] + golden, 1)
    finally:
        if not args.keep_indexes:
            for name in idx.values():
                assert name.startswith(PREFIX)
                try:
                    if pc.indexes.exists(name=name):
                        pc.indexes.delete(name=name)
                        print("deleted index", name)
                except Exception as exc:
                    print("could not delete", name, exc)
        with open(os.path.join(args.scratch, "smoke_results.json"), "w") as f:
            json.dump({"run_id": run_id, "indexes": idx, "results": r.results}, f, indent=2, ensure_ascii=False)

    failed = [x for x in r.results if not x["ok"]]
    print(f"\n{len(r.results) - len(failed)}/{len(r.results)} steps as expected")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
