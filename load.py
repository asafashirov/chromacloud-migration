#!/usr/bin/env python3
"""Stream documents.jsonl into a fresh Pinecone namespace in bounded batches.

Batches are size-aware: a batch is flushed before it would exceed
--max-docs (at most 1,000) or --max-request-bytes of encoded JSON (default
1,900,000, under the 2 MB request limit). Only one batch is in memory.

Upserts are idempotent by _id, so a rerun overwrites the same documents with
the same content. The loader refuses a namespace that already holds records
unless load_state.json shows an earlier load of the same documents.jsonl into
the same index and namespace (a rerun). --resume skips the lines an earlier
run already confirmed.

Exit codes: 0 = every document accepted; 1 = an upsert failed or the index
schema does not match; 2 = usage error (missing/failed convert output,
namespace not empty).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import orjson

import miglib as m


def upsert_with_retry(index, namespace: str, batch: list[dict], max_attempts: int) -> int:
    from pinecone.errors.exceptions import (
        PineconeConnectionError,
        PineconeTimeoutError,
        RateLimitError,
        ServiceError,
    )

    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            return index.documents.upsert(namespace=namespace, documents=batch).upserted_count
        except (RateLimitError, ServiceError, PineconeConnectionError, PineconeTimeoutError) as exc:
            if attempt == max_attempts:
                raise
            m.LOG.warning("upsert attempt %d failed (%s); retrying in %.0fs", attempt, type(exc).__name__, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--workdir", default="migration_work")
    ap.add_argument("--max-docs", type=int, default=m.MAX_DOCS_PER_UPSERT)
    ap.add_argument("--max-request-bytes", type=int, default=1_900_000)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-metric-mismatch", action="store_true",
                    help="testing only: load even if the index metric differs from the source")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    m.setup_logging(args.verbose)
    from pinecone.errors.exceptions import NotFoundError

    if not 1 <= args.max_docs <= m.MAX_DOCS_PER_UPSERT:
        m.LOG.error("--max-docs must be 1..%d", m.MAX_DOCS_PER_UPSERT)
        return m.EXIT_USAGE
    if args.max_request_bytes > m.MAX_REQUEST_BYTES:
        m.LOG.error("--max-request-bytes must be <= %d", m.MAX_REQUEST_BYTES)
        return m.EXIT_USAGE
    summary_path = m.wpath(args.workdir, m.CONVERT_SUMMARY)
    docs_path = m.wpath(args.workdir, m.DOCUMENTS_FILE)
    if not os.path.exists(summary_path) or not m.read_json(summary_path).get("ok") or not os.path.exists(docs_path):
        m.LOG.error("convert.py did not finish successfully; nothing to load")
        return m.EXIT_USAGE
    summary = m.read_json(summary_path)
    docs_sha = m.sha256_file(docs_path)
    if docs_sha != summary.get("documents_sha256"):
        m.LOG.error("documents.jsonl does not match convert_summary.json")
        return m.EXIT_USAGE
    cfg, fm = m.load_config(args.workdir)
    expected_metric = fm.get("pinecone_metric")

    pc = m.get_pinecone()
    desc, index = m.open_pinecone_index(pc, args.index)
    fields = m.schema_fields(desc)
    dense = fields.get(cfg.vector_field)
    if dense is None or getattr(dense, "dimension", None) is None:
        m.LOG.error("index %s has no dense_vector field %r", args.index, cfg.vector_field)
        return m.EXIT_FAIL
    if cfg.dimension is not None and dense.dimension != cfg.dimension:
        m.LOG.error("dimension mismatch: index %d, source %d", dense.dimension, cfg.dimension)
        return m.EXIT_FAIL
    if expected_metric and dense.metric != expected_metric:
        msg = f"metric mismatch: index {dense.metric}, source {fm['source_metric']['space']} -> {expected_metric}"
        if not args.allow_metric_mismatch:
            m.LOG.error(msg)
            return m.EXIT_FAIL
        m.LOG.warning("%s (continuing because of --allow-metric-mismatch)", msg)
    if cfg.text_mode == "fts" and cfg.text_field not in fields:
        m.LOG.error("documents were converted with --text-mode fts but the index does not declare %r",
                    cfg.text_field)
        return m.EXIT_FAIL

    state_path = m.wpath(args.workdir, m.LOAD_STATE)
    prior = m.read_json(state_path) if os.path.exists(state_path) else None
    same_target = bool(prior) and prior["index"] == args.index and prior["namespace"] == args.namespace \
        and prior["documents_sha256"] == docs_sha
    try:
        existing = index.describe_namespace(name=args.namespace).record_count
    except NotFoundError:
        existing = 0
    if existing and not same_target:
        m.LOG.error("namespace %r already holds %d records and load_state.json does not show a load of "
                    "this documents.jsonl into it; use a fresh namespace", args.namespace, existing)
        return m.EXIT_USAGE
    skip = prior.get("lines_done", 0) if (args.resume and same_target) else 0
    if existing:
        m.LOG.info("rerun into %r (%d records present); upserts overwrite by _id", args.namespace, existing)

    state = {"index": args.index, "namespace": args.namespace, "documents_sha256": docs_sha,
             "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "lines_done": skip,
             "complete": False, "runs": (prior.get("runs", 0) + 1) if same_target else 1}
    m.write_json(state_path, state)

    envelope = len(b'{"documents":[]}')
    batch: list[dict] = []
    batch_bytes = envelope
    sent = batches = max_batch_bytes = 0
    started = time.time()

    def flush(done_through_line: int) -> None:
        nonlocal batch, batch_bytes, sent, batches, max_batch_bytes
        if not batch:
            return
        n = upsert_with_retry(index, args.namespace, batch, args.max_attempts)
        if n != len(batch):
            raise RuntimeError(f"upserted_count {n} != batch size {len(batch)}")
        sent += n
        batches += 1
        max_batch_bytes = max(max_batch_bytes, batch_bytes)
        state["lines_done"] = done_through_line
        m.write_json(state_path, state)
        if batches % 10 == 0:
            m.LOG.info("%d documents in %d batches", sent, batches)
        batch, batch_bytes = [], envelope

    line_no = 0
    try:
        with open(docs_path, encoding="utf-8") as f:
            for raw in f:
                line_no += 1
                if line_no <= skip:
                    continue
                doc = json.loads(raw)
                size = len(orjson.dumps(doc))
                if size + envelope > args.max_request_bytes:
                    m.LOG.error("document %r is %d bytes; it cannot fit one request", doc["_id"], size)
                    return m.EXIT_FAIL
                if len(batch) >= args.max_docs or batch_bytes + size + 1 > args.max_request_bytes:
                    flush(line_no - 1)
                batch.append(doc)
                batch_bytes += size + (1 if len(batch) > 1 else 0)
            flush(line_no)
    except Exception as exc:
        m.LOG.error("upsert failed after line %d: %s", state["lines_done"], exc)
        print(f"LOAD FAILED: {exc}. Fix the cause and rerun (add --resume to skip confirmed lines).")
        return m.EXIT_FAIL

    state.update({"complete": True, "documents_sent": sent, "batches": batches,
                  "max_batch_bytes": max_batch_bytes, "seconds": round(time.time() - started, 2),
                  "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    m.write_json(state_path, state)
    print(f"LOAD OK: {sent} documents in {batches} batches (largest {max_batch_bytes} bytes) "
          f"-> {args.index}/{args.namespace}. Run validate.py next; counts lag writes.")
    return m.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
