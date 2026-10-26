#!/usr/bin/env python3
"""Create (or verify) the Pinecone document index for the migration.

Reads dimension and metric from preflight.json (Chroma l2 -> euclidean,
cosine -> cosine, ip -> dotproduct). The schema declares one dense_vector
field. With --full-text-search it also declares the text field with
full_text_search, which lifts the 40 KB metadata cap on document text (100 KB
per field instead); the schema cannot be changed after creation.

If the index already exists, its schema is checked against the expected one.

Exit codes: 0 = index ready with the expected schema; 1 = an existing index
has a different schema; 2 = usage error (no preflight report, no dimension).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import miglib as m


def expected_schema(args, dimension: int, metric: str) -> dict:
    from pinecone import SchemaBuilder

    sb = SchemaBuilder().add_dense_vector_field(name=args.vector_field, dimension=dimension, metric=metric)
    if args.full_text_search:
        sb = sb.add_string_field(name=args.text_field, full_text_search={"language": args.language})
    return sb.build()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="index name: lowercase letters, digits, '-', max 45")
    ap.add_argument("--workdir", default="migration_work")
    ap.add_argument("--dimension", type=int, help="override preflight dimension")
    ap.add_argument("--metric", choices=["cosine", "euclidean", "dotproduct"], help="override preflight metric")
    ap.add_argument("--vector-field", default=m.DEFAULT_VECTOR_FIELD)
    ap.add_argument("--text-field", default=m.DEFAULT_TEXT_FIELD)
    ap.add_argument("--full-text-search", action="store_true")
    ap.add_argument("--language", default="en")
    ap.add_argument("--cloud", default="aws")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--deletion-protection", choices=["enabled", "disabled"], default="disabled")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    m.setup_logging(args.verbose)

    pre_path = m.wpath(args.workdir, m.PREFLIGHT_REPORT)
    pre = m.read_json(pre_path) if os.path.exists(pre_path) else {}
    dimension = args.dimension or pre.get("dimension")
    metric = args.metric or (pre.get("metric") or {}).get("pinecone_metric")
    if not dimension or not metric:
        m.LOG.error("need dimension and metric: run preflight.py or pass --dimension/--metric")
        return m.EXIT_USAGE
    if pre and args.metric and args.metric != pre["metric"]["pinecone_metric"]:
        m.LOG.warning("metric override %s differs from the source collection (%s -> %s)",
                      args.metric, pre["metric"]["space"], pre["metric"]["pinecone_metric"])

    schema = expected_schema(args, dimension, metric)
    pc = m.get_pinecone()
    if pc.indexes.exists(name=args.index):
        desc = pc.indexes.describe(name=args.index)
        fields = m.schema_fields(desc)
        dense = fields.get(args.vector_field)
        problems = []
        if dense is None or getattr(dense, "dimension", None) != dimension:
            problems.append(f"dense field {args.vector_field!r}: expected dimension {dimension}, got {dense}")
        elif getattr(dense, "metric", None) != metric:
            problems.append(f"dense field metric: expected {metric}, got {dense.metric}")
        if args.full_text_search and args.text_field not in fields:
            problems.append(f"text field {args.text_field!r} is not declared with full_text_search")
        if problems:
            for p in problems:
                print("INDEX MISMATCH:", p)
            return m.EXIT_FAIL
        print(f"index {args.index} exists with the expected schema: {schema}")
        return m.EXIT_OK

    pc.indexes.create(
        name=args.index,
        schema=schema,
        deployment={"deployment_type": "managed", "cloud": args.cloud, "region": args.region},
        deletion_protection=args.deletion_protection,
        timeout=-1,
    )
    while not pc.indexes.describe(name=args.index).status.ready:
        time.sleep(2)
    print(f"INDEX READY {args.index}: {schema}")
    return m.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
