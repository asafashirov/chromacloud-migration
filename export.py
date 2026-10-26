#!/usr/bin/env python3
"""Export one Chroma collection to JSONL, streaming page by page.

Each line is {"id", "embedding", "document", "metadata"} exactly as Chroma
returns them. Only the set of IDs is held in memory (to prove uniqueness).
Run this after writes are paused.

Outputs (in --workdir): export.jsonl, export_manifest.json
Exit codes: 0 = unique exported IDs == collection.count() and the count did
not change during export; 1 = mismatch, duplicates or a count change
(manifest written with ok=false); 2 = usage or connection error.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import miglib as m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m.add_chroma_args(ap)
    ap.add_argument("--workdir", default="migration_work")
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    m.setup_logging(args.verbose)
    os.makedirs(args.workdir, exist_ok=True)

    try:
        col = m.get_chroma_collection(args)
    except Exception as exc:
        m.LOG.error("cannot open collection %r: %s", args.collection, exc)
        return m.EXIT_USAGE

    count_before = col.count()
    metric = m.detect_metric(col)
    out_path = m.wpath(args.workdir, m.EXPORT_FILE)
    tmp_path = out_path + ".partial"
    seen: set[str] = set()
    duplicates: list[str] = []
    rows = 0
    dims: dict[int, int] = {}
    started = time.time()
    with open(tmp_path, "w", encoding="utf-8") as f:
        for page in m.iter_chroma_pages(col, args.page_size, ["embeddings", "documents", "metadatas"]):
            for rec in page:
                rows += 1
                if rec["id"] in seen:
                    duplicates.append(rec["id"])
                seen.add(rec["id"])
                if rec["embedding"] is not None:
                    dims[len(rec["embedding"])] = dims.get(len(rec["embedding"]), 0) + 1
                f.write(m.dumps(rec) + "\n")
            m.LOG.info("exported %d rows", rows)
    os.replace(tmp_path, out_path)
    count_after = col.count()

    problems = []
    if duplicates:
        problems.append(f"{len(duplicates)} duplicate IDs while paging (e.g. {duplicates[:3]})")
    if count_after != count_before:
        problems.append(f"count changed during export: {count_before} -> {count_after}; writes are not paused")
    if len(seen) != count_before:
        problems.append(f"unique exported IDs {len(seen)} != collection.count() {count_before}")

    manifest = {
        "collection": args.collection,
        "source": "local:" + args.chroma_path if args.chroma_path else "chroma-cloud",
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "count_before": count_before,
        "count_after": count_after,
        "rows": rows,
        "unique_ids": len(seen),
        "page_size": args.page_size,
        "dimensions": dims,
        "metric": metric,
        "file": m.EXPORT_FILE,
        "sha256": m.sha256_file(out_path),
        "seconds": round(time.time() - started, 2),
        "problems": problems,
        "ok": not problems,
    }
    m.write_json(m.wpath(args.workdir, m.EXPORT_MANIFEST), manifest)
    print(f"count={count_before} rows={rows} unique_ids={len(seen)} -> {out_path}")
    if problems:
        for p in problems:
            print("EXPORT FAILED:", p)
        return m.EXIT_FAIL
    print("EXPORT OK")
    return m.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
