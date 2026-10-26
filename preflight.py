#!/usr/bin/env python3
"""Preflight: inspect one Chroma collection before migrating it to Pinecone.

Reports count, embedding dimension (and whether it is consistent), distance
metric, metadata keys and value types, and ID validity for Pinecone, then
classifies every finding as blocking, transform, warning or info.

Exit codes: 0 = no blocking issues (transforms allowed or none needed),
1 = blocking issues, or transforms needed without --allow-transforms,
2 = usage or connection error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import miglib as m
from filters import array_keys_from_kinds, translate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m.add_chroma_args(ap)
    m.add_convert_policy_args(ap)
    ap.add_argument("--workdir", default="migration_work")
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--allow-transforms", action="store_true",
                    help="Accept the documented convert.py transforms (ID mapping, key renames, "
                    "None drops, and array/int stringify when that policy is chosen).")
    ap.add_argument("--app-filters",
                    help="JSON file: list of {name, where} / {name, where_document} the app uses")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    m.setup_logging(args.verbose)
    os.makedirs(args.workdir, exist_ok=True)

    try:
        col = m.get_chroma_collection(args)
    except Exception as exc:
        m.LOG.error("cannot open collection %r: %s", args.collection, exc)
        return m.EXIT_USAGE

    cfg = m.config_from_args(args)
    count = col.count()
    metric = m.detect_metric(col)
    an = m.Analyzer(cfg)
    for page in m.iter_chroma_pages(col, args.page_size, ["embeddings", "documents", "metadatas"]):
        for rec in page:
            an.add(rec)
    issues = an.issues(metric)
    if an.unique_ids() != count:
        issues.insert(0, {"code": "count_mismatch", "severity": "blocking",
                          "message": f"collection.count()={count} but the scan saw {an.unique_ids()} unique IDs "
                          f"({an.count} rows). Pause writes and rerun.", "count": None, "examples": []})

    filter_results = []
    if args.app_filters:
        cfg.field_map = an.field_map()
        strf = {k for k, c in an.key_kinds.items()
                if cfg.array_policy == "stringify" and any(c.get(x) for x in m.NON_STRING_ARRAY_KINDS)}
        arrays = array_keys_from_kinds({k: dict(c) for k, c in an.key_kinds.items()})
        for entry in m.read_json(args.app_filters):
            filter_results.append(translate(entry, cfg, strf, arrays))
        redesign = [f["name"] for f in filter_results if f["status"] == "needs_redesign"]
        verify = [f["name"] for f in filter_results if f["status"] == "translated_verify"]
        if redesign:
            issues.append({"code": "filters_need_redesign", "severity": "warning",
                           "message": "App filters with no direct Pinecone equivalent", "count": len(redesign),
                           "examples": redesign})
        if verify:
            issues.append({"code": "filters_need_verification", "severity": "warning",
                           "message": "Array filters rewritten ($contains -> $in, $not_contains -> $nin); "
                           "run validate.py --app-filters to compare results", "count": len(verify),
                           "examples": verify})

    blocking = [i for i in issues if i["severity"] == "blocking"]
    transforms = [i for i in issues if i["severity"] == "transform"]
    ok = not blocking and (args.allow_transforms or not transforms)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "collection": args.collection,
        "source": "local:" + args.chroma_path if args.chroma_path else "chroma-cloud",
        "count": count,
        "scanned_rows": an.count,
        "unique_ids": an.unique_ids(),
        "dimension": an.dimension(),
        "dimensions_seen": dict(an.dims),
        "metric": metric,
        "embedding_function": m.embedding_function_info(col),
        "documents": {"none": an.docs_none, "empty": an.docs_empty, "non_ascii": an.docs_non_ascii,
                      "max_bytes": an.max_doc_bytes},
        "records_without_metadata": an.metadata_none,
        "metadata_keys": an.key_report(),
        "ids": {"invalid": len(an.invalid_ids), "reasons": dict(an.invalid_id_reasons)},
        "policies": cfg.to_json(),
        "allow_transforms": args.allow_transforms,
        "issues": issues,
        "app_filters": filter_results,
        "ok": ok,
    }
    out = m.wpath(args.workdir, m.PREFLIGHT_REPORT)
    m.write_json(out, report)

    print(f"collection {args.collection}: count={count} dimension={report['dimension']} "
          f"metric={metric['space']} (from {metric['source']}) -> pinecone {metric['pinecone_metric']}")
    print(f"metadata keys: {len(an.key_kinds)}  invalid IDs: {len(an.invalid_ids)}  "
          f"documents None/empty: {an.docs_none}/{an.docs_empty}")
    for i in issues:
        ex = f" e.g. {[m.short(x) for x in i['examples'][:3]]}" if i["examples"] else ""
        cnt = f" ({i['count']})" if i["count"] is not None else ""
        print(f"[{i['severity'].upper()}] {i['code']}{cnt}: {i['message']}{ex}")
    for f in filter_results:
        print(f"[FILTER {f['status']}] {f['name']}: {json.dumps(f['pinecone_filter'])} {'; '.join(f['notes'])}")
    print(f"report: {out}")
    if blocking:
        print(f"PREFLIGHT FAILED: {len(blocking)} blocking issue(s)")
        return m.EXIT_FAIL
    if transforms and not args.allow_transforms:
        print(f"PREFLIGHT FAILED: {len(transforms)} issue(s) need convert.py transforms; "
              "review them and rerun with --allow-transforms")
        return m.EXIT_FAIL
    print("PREFLIGHT OK")
    return m.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
