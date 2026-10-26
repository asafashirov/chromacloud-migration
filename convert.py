#!/usr/bin/env python3
"""Convert export.jsonl into Pinecone documents, with mapping files.

Pass 1 streams the export to collect keys and IDs and builds two tables:
  field_map.json  every metadata key -> Pinecone field (renames are
                  collision-free, deterministic and recorded with a reason)
  id_map.jsonl    every ID Pinecone rejects -> prefix + sha256(ID)
Pass 2 streams the export again and writes documents.jsonl, one Pinecone
document per line, plus transforms.jsonl, one line per per-record transform
or drop (ID remap, None dropped, document omitted, array/int stringified).

Defaults: numeric/boolean arrays and ints beyond 2^53 are rejected
(--array-policy / --big-int-policy stringify to convert them). Nothing is
dropped silently. If any record cannot be converted, no documents.jsonl is
written.

Exit codes: 0 = every record converted; 1 = blocking issues (see
convert_summary.json); 2 = usage error or missing/failed export.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import miglib as m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m.add_convert_policy_args(ap)
    ap.add_argument("--workdir", default="migration_work")
    ap.add_argument("--max-errors-shown", type=int, default=20)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    m.setup_logging(args.verbose)

    manifest_path = m.wpath(args.workdir, m.EXPORT_MANIFEST)
    export_path = m.wpath(args.workdir, m.EXPORT_FILE)
    if not os.path.exists(manifest_path) or not os.path.exists(export_path):
        m.LOG.error("run export.py first (missing %s or %s)", manifest_path, export_path)
        return m.EXIT_USAGE
    manifest = m.read_json(manifest_path)
    if not manifest.get("ok"):
        m.LOG.error("export manifest is not ok: %s", manifest.get("problems"))
        return m.EXIT_USAGE
    if m.sha256_file(export_path) != manifest["sha256"]:
        m.LOG.error("export.jsonl does not match the manifest checksum")
        return m.EXIT_USAGE

    cfg = m.config_from_args(args)
    an = m.Analyzer(cfg)
    for rec in m.iter_jsonl(export_path):
        an.add(rec)
    cfg.dimension = an.dimension()
    cfg.field_map = an.field_map()
    cfg.id_map = an.id_map()
    issues = an.issues(manifest.get("metric"))
    blocking = [i for i in issues if i["severity"] == "blocking"]

    docs_path = m.wpath(args.workdir, m.DOCUMENTS_FILE)
    tmp_docs = docs_path + ".partial"
    tr_path = m.wpath(args.workdir, m.TRANSFORMS_FILE)
    tmp_tr = tr_path + ".partial"
    errors: list[dict] = []
    error_counts: Counter = Counter()
    transform_counts: Counter = Counter()
    written = 0
    max_doc_bytes = 0
    with open(tmp_docs, "w", encoding="utf-8") as fd, open(tmp_tr, "w", encoding="utf-8") as ft:
        for rec in m.iter_jsonl(export_path):
            doc, transforms, errs = m.convert_record(rec, cfg)
            for t in transforms:
                transform_counts[t["kind"]] += 1
                ft.write(m.dumps(t) + "\n")
            for e in errs:
                error_counts[e["code"]] += 1
                if len(errors) < 1000:
                    errors.append(e)
            if doc is not None:
                line = m.dumps(doc)
                max_doc_bytes = max(max_doc_bytes, len(line.encode("utf-8")))
                fd.write(line + "\n")
                written += 1

    rename_counts = Counter()
    for k, c in an.key_kinds.items():
        if cfg.field_map[k]["reason"]:
            rename_counts[k] = sum(c.values())
    fields = {
        k: {**v, "value_kinds": dict(an.key_kinds[k]), "records": sum(an.key_kinds[k].values())}
        for k, v in cfg.field_map.items()
    }
    ok = not blocking and not error_counts and written == an.count
    m.write_json(m.wpath(args.workdir, m.FIELD_MAP_FILE), {
        "config": cfg.to_json(),
        "source_metric": manifest.get("metric"),
        "pinecone_metric": (manifest.get("metric") or {}).get("pinecone_metric"),
        "fields": fields,
    })
    with open(m.wpath(args.workdir, m.ID_MAP_FILE), "w", encoding="utf-8") as f:
        for sid, pid in sorted(cfg.id_map.items()):
            f.write(m.dumps({"source_id": sid, "pinecone_id": pid, "reason": m.id_problem(sid)}) + "\n")
    summary = {
        "converted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "export_sha256": manifest["sha256"],
        "records_in": an.count,
        "documents_out": written if ok else 0,
        "dimension": cfg.dimension,
        "transform_counts": dict(transform_counts),
        "renamed_keys": {k: cfg.field_map[k]["dest"] for k in rename_counts},
        "renamed_key_record_counts": dict(rename_counts),
        "remapped_ids": len(cfg.id_map),
        "records_without_metadata": an.metadata_none,
        "max_document_bytes": max_doc_bytes,
        "error_counts": dict(error_counts),
        "errors": errors,
        "issues": issues,
        "ok": ok,
    }
    if ok:
        os.replace(tmp_docs, docs_path)
        os.replace(tmp_tr, tr_path)
        if os.path.exists(tr_path + ".failed"):
            os.remove(tr_path + ".failed")
        summary["documents_sha256"] = m.sha256_file(docs_path)
    else:
        os.remove(tmp_docs)
        os.replace(tmp_tr, tr_path + ".failed")
        if os.path.exists(docs_path):
            os.remove(docs_path)
    m.write_json(m.wpath(args.workdir, m.CONVERT_SUMMARY), summary)

    print(f"records={an.count} documents={summary['documents_out']} remapped_ids={len(cfg.id_map)} "
          f"renamed_keys={len(rename_counts)}")
    for k, dest in summary["renamed_keys"].items():
        print(f"  rename {k!r} -> {dest!r} ({cfg.field_map[k]['reason']}, {rename_counts[k]} records)")
    for kind, n in sorted(transform_counts.items()):
        print(f"  transform {kind}: {n}")
    if not ok:
        for i in blocking:
            print(f"[BLOCKING] {i['code']}: {i['message']}")
        for code, n in error_counts.items():
            print(f"[ERROR] {code}: {n} record(s)")
        for e in errors[: args.max_errors_shown]:
            print("   ", {k: (v[:80] if isinstance(v, str) else v) for k, v in e.items()})
        print("CONVERT FAILED: no documents.jsonl written")
        return m.EXIT_FAIL
    print(f"CONVERT OK -> {docs_path}")
    return m.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
