#!/usr/bin/env python3
"""Validate a Path A load against the (write-paused) Chroma collection.

Checks, in order:
  source_unchanged      collection.count() still equals the exported count
  metric_and_dimension  live Chroma space and dimension match the index schema
  namespace_count       polls describe_namespace until record_count matches
  id_set                full ID-set reconciliation (Chroma IDs mapped through
                        id_map.jsonl vs every ID listed in the namespace)
  sample_fields         field-by-field comparison of a sample of fetched
                        documents with the source records, converted with the
                        recorded field_map.json / id_map.jsonl, plus a
                        round trip back to the Chroma record
  search_visible        polls until probe searches return top_k results
  golden_queries        top-k overlap between Chroma and Pinecone for the same
                        query embeddings (tie-aware; retried until
                        --search-timeout to absorb freshness lag)
  self_retrieval        diagnostics only (never fails): does each sampled
                        vector find its own record?
  app_filters           optional: Chroma `where` vs translated Pinecone
                        `filter`, compared as ID sets

Golden queries: pass --golden-queries FILE (JSONL lines with "query_id" and
"embedding", produced with the embedding model your app uses), or for tests
--golden-from-sample N (perturbed copies of stored vectors).

Exit codes: 0 = every check passed; 1 = at least one check failed;
2 = usage error (missing workdir files, missing golden queries).
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import defaultdict

import miglib as m
from filters import array_keys_from_kinds, translate


def same_value(e, a) -> bool:
    if isinstance(e, bool) or isinstance(a, bool):
        return type(e) is bool and type(a) is bool and e == a
    if isinstance(e, (int, float)) and isinstance(a, (int, float)):
        return e == a or math.isclose(e, a, rel_tol=1e-12)
    if isinstance(e, list) and isinstance(a, list):
        return len(e) == len(a) and all(same_value(x, y) for x, y in zip(e, a))
    return e == a


def same_vector(e, a, rel=1e-6) -> bool:
    if not isinstance(e, list) or not isinstance(a, list) or len(e) != len(a):
        return False
    return all(math.isclose(x, y, rel_tol=rel, abs_tol=1e-7) for x, y in zip(e, a))


def diff_docs(expected: dict, actual: dict, vector_field: str) -> list[str]:
    out = []
    ek, ak = set(expected), set(actual) - {"_score"}
    for k in sorted(ek - ak):
        out.append(f"missing field {k!r}")
    for k in sorted(ak - ek):
        out.append(f"unexpected field {k!r}")
    for k in sorted(ek & ak):
        if k == vector_field:
            if not same_vector(expected[k], actual[k]):
                out.append("vector differs")
        elif not same_value(expected[k], actual[k]):
            out.append(f"{k!r}: expected {str(expected[k])[:60]!r}, got {str(actual[k])[:60]!r}")
    return out


def diff_roundtrip(src: dict, back: dict) -> list[str]:
    out = []
    if src["id"] != back["id"]:
        out.append("id does not round-trip")
    if src.get("document") != back.get("document"):
        out.append("document does not round-trip")
    if not same_vector(src["embedding"], back["embedding"]):
        out.append("vector does not round-trip")
    sm, bm = src.get("metadata") or {}, back.get("metadata") or {}
    if set(sm) != set(bm):
        out.append(f"metadata keys differ: {sorted(set(sm) ^ set(bm))}")
    for k in set(sm) & set(bm):
        if not same_value(sm[k], bm[k]):
            out.append(f"metadata {k!r} does not round-trip")
    return out


def chroma_distance_to_score(d: float, space: str) -> float:
    return d if space == "l2" else 1.0 - d


def close_score(x: float, y: float) -> bool:
    return abs(x - y) <= 1e-3 + 0.02 * max(abs(x), abs(y))


class Report:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, name: str, status: str, summary: str, **details):
        self.checks.append({"check": name, "status": status, "summary": summary, **details})
        print(f"[{status.upper():4}] {name}: {summary}")

    @property
    def failed(self) -> bool:
        return any(c["status"] == "fail" for c in self.checks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m.add_chroma_args(ap)
    ap.add_argument("--index", required=True)
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--workdir", default="migration_work")
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--self-retrieval", type=int, default=50, help="sampled records to self-query")
    ap.add_argument("--golden-queries", help="JSONL: {query_id, embedding} per line")
    ap.add_argument("--golden-from-sample", type=int, default=0)
    ap.add_argument("--golden-noise", type=float, default=0.05)
    ap.add_argument("--min-mean-overlap", type=float, default=0.9)
    ap.add_argument("--min-query-overlap", type=float, default=0.6)
    ap.add_argument("--app-filters", help="JSON list of {name, where} / {name, where_document}")
    ap.add_argument("--poll-timeout", type=float, default=600)
    ap.add_argument("--poll-interval", type=float, default=5)
    ap.add_argument("--search-timeout", type=float, default=180,
                    help="seconds to keep retrying search-based checks while new writes become searchable")
    ap.add_argument("--dimension", type=int,
                    help="expected index dimension when the source collection is empty (otherwise unchecked)")
    ap.add_argument("--report", help="report path (default: <workdir>/validation_report.json)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    m.setup_logging(args.verbose)
    from pinecone.errors.exceptions import NotFoundError

    try:
        manifest = m.read_json(m.wpath(args.workdir, m.EXPORT_MANIFEST))
        summary = m.read_json(m.wpath(args.workdir, m.CONVERT_SUMMARY))
        cfg, fm = m.load_config(args.workdir)
    except FileNotFoundError as exc:
        m.LOG.error("missing workdir file: %s", exc)
        return m.EXIT_USAGE
    if not summary.get("ok"):
        m.LOG.error("convert_summary.json is not ok")
        return m.EXIT_USAGE
    expected_count = manifest["count_before"]
    if expected_count and not (args.golden_queries or args.golden_from_sample):
        m.LOG.error("pass --golden-queries FILE (embeddings from your app's model) or --golden-from-sample N")
        return m.EXIT_USAGE

    try:
        col = m.get_chroma_collection(args)
    except Exception as exc:
        m.LOG.error("cannot open collection %r: %s", args.collection, exc)
        return m.EXIT_USAGE
    pc = m.get_pinecone()
    index_model, index = m.open_pinecone_index(pc, args.index)
    rep = Report()

    now = col.count()
    rep.add("source_unchanged", "pass" if now == expected_count else "fail",
            f"collection.count()={now}, exported={expected_count}"
            + ("" if now == expected_count else " (source changed after export: writes not paused?)"))

    live_metric = m.detect_metric(col)
    fields = m.schema_fields(index_model)
    dense = fields.get(cfg.vector_field)
    problems = []
    if live_metric["space"] != manifest["metric"]["space"]:
        problems.append(f"Chroma space changed: {manifest['metric']['space']} -> {live_metric['space']}")
    want = m.METRIC_MAP.get(live_metric["space"])
    if dense is None:
        problems.append(f"index has no field {cfg.vector_field!r}")
    else:
        if dense.metric != want:
            problems.append(f"index metric {dense.metric} but Chroma space {live_metric['space']} needs {want}")
        expected_dim = cfg.dimension if cfg.dimension is not None else args.dimension
        if expected_dim is None:
            dim_note = (f"dimension not checked: the source collection is empty and --dimension was not "
                        f"given (index has {dense.dimension})")
        elif dense.dimension != expected_dim:
            problems.append(f"index dimension {dense.dimension} != expected {expected_dim}")
        else:
            dim_note = f"dimension {expected_dim}" + ("" if cfg.dimension is not None else " (from --dimension)")
    dim_unchecked = dense is not None and cfg.dimension is None and args.dimension is None
    rep.add("metric_and_dimension", "fail" if problems else ("warn" if dim_unchecked else "pass"),
            "; ".join(problems) or f"{live_metric['space']} -> {want}, {dim_note}")

    deadline = time.time() + args.poll_timeout
    while True:
        try:
            dest_count = index.describe_namespace(name=args.namespace).record_count
        except NotFoundError:
            dest_count = 0
        if dest_count == expected_count or time.time() > deadline:
            break
        m.LOG.info("namespace count %d, waiting for %d", dest_count, expected_count)
        time.sleep(args.poll_interval)
    rep.add("namespace_count", "pass" if dest_count == expected_count else "fail",
            f"record_count={dest_count}, expected {expected_count}")

    source_ids: list[str] = []
    for page in m.iter_chroma_pages(col, args.page_size, []):
        source_ids.extend(r["id"] for r in page)
    expected_ids = {cfg.id_map.get(s, s) for s in source_ids}
    missing: set[str] = set()
    extra: set[str] = set()
    dest_ids: set[str] = set()
    for _ in range(4):
        try:
            dest_ids = {d.id for d in index.documents.list(namespace=args.namespace, limit=100)}
        except NotFoundError:
            dest_ids = set()
        missing, extra = expected_ids - dest_ids, dest_ids - expected_ids
        if not missing and not extra:
            break
        time.sleep(args.poll_interval)
    rep.add("id_set", "fail" if (missing or extra or len(source_ids) != expected_count) else "pass",
            f"source {len(expected_ids)} ids, destination {len(dest_ids)}, missing {len(missing)}, extra {len(extra)}",
            missing_examples=sorted(missing)[:10], extra_examples=sorted(extra)[:10])

    if not source_ids:
        rep.add("sample_fields", "skip", "empty collection")
        rep.add("golden_queries", "skip", "empty collection")
    else:
        rng = random.Random(args.seed)
        sample = rng.sample(sorted(source_ids), min(args.sample, len(source_ids)))
        transformed: dict[str, list] = defaultdict(list)
        for t in m.iter_jsonl(m.wpath(args.workdir, m.TRANSFORMS_FILE)):
            transformed[t["source_id"]].append(t)
        in_sample = set(sample)
        extra_sample = [s for s in sorted(transformed) if s not in in_sample][: args.sample]
        sample_ids = sample + extra_sample

        src = m.chroma_get_records(col, sample_ids)
        pids = [cfg.id_map.get(s, s) for s in sample_ids]
        actual: dict[str, dict] = {}
        for i in range(0, len(pids), 100):
            res = index.documents.fetch(namespace=args.namespace, ids=pids[i : i + 100])
            actual.update({k: v.to_dict() for k, v in res.documents.items()})
        mismatches = {}
        for sid in sample_ids:
            rec = src.get(sid)
            pid = cfg.id_map.get(sid, sid)
            if rec is None:
                mismatches[sid] = ["missing in Chroma"]
                continue
            expected, _, errs = m.convert_record(rec, cfg)
            if errs:
                mismatches[sid] = [f"source record no longer converts: {errs}"]
                continue
            got = actual.get(pid)
            if got is None:
                mismatches[sid] = ["missing in Pinecone fetch"]
                continue
            d = diff_docs(expected, got, cfg.vector_field)
            d += diff_roundtrip(rec, m.reverse_document(got, cfg, transformed.get(sid, [])))
            if d:
                mismatches[sid] = d
        rep.add("sample_fields", "fail" if mismatches else "pass",
                f"{len(sample_ids)} records compared ({len(extra_sample)} added because they were transformed), "
                f"{len(mismatches)} mismatched",
                mismatches={k[:80]: v for k, v in list(mismatches.items())[:20]})

        space = live_metric["space"]
        k = min(args.top_k, len(source_ids))
        probe = [s for s in sample_ids if s in src]

        def search(vec, include=None):
            return index.documents.search(namespace=args.namespace, top_k=k, include_fields=include,
                                          score_by=[{"type": "dense_vector", "fields": [cfg.vector_field],
                                                     "values": vec}])

        search_deadline = time.time() + args.search_timeout
        while True:
            visible = [len(search(src[s]["embedding"]).matches) for s in probe[:3]]
            if all(v == k for v in visible) or time.time() > search_deadline:
                break
            m.LOG.info("search returns %s of %d results; waiting for the index to become searchable", visible, k)
            time.sleep(args.poll_interval)
        rep.add("search_visible", "pass" if all(v == k for v in visible) else "fail",
                f"probe searches returned {visible} of top_k={k}")

        queries = []
        if args.golden_queries:
            queries = [(q.get("query_id", str(i)), q["embedding"])
                       for i, q in enumerate(m.iter_jsonl(args.golden_queries))]
        if args.golden_from_sample:
            for sid in rng.sample(sorted(src), min(args.golden_from_sample, len(src))):
                v = src[sid]["embedding"]
                norm = math.sqrt(sum(x * x for x in v)) or 1.0
                scale = args.golden_noise * norm / math.sqrt(len(v))
                queries.append((f"noisy:{sid[:40]}", [x + rng.gauss(0, scale) for x in v]))
        chroma_results = []
        for qid, qvec in queries:
            cr = col.query(query_embeddings=[qvec], n_results=k, include=["distances"])
            chroma_results.append(([cfg.id_map.get(x, x) for x in cr["ids"][0]],
                                   [chroma_distance_to_score(d, space) for d in cr["distances"][0]]))

        def run_golden():
            per_query = []
            for (qid, qvec), (c_ids, c_scores) in zip(queries, chroma_results):
                pr = search(qvec)
                p_ids = [x.id for x in pr.matches]
                p_scores = [x.score for x in pr.matches]
                denom = max(1, min(k, len(c_ids)))
                overlap = len(set(c_ids) & set(p_ids))
                ties = 0
                if p_scores:
                    for cid, cs in zip(c_ids, c_scores):
                        if cid not in p_ids and close_score(cs, p_scores[-1]):
                            ties += 1
                per_query.append({"query_id": qid, "overlap": overlap / denom,
                                  "tie_adjusted": min(1.0, (overlap + ties) / denom),
                                  "top1_same": bool(c_ids and p_ids and c_ids[0] == p_ids[0]),
                                  "top1_score_close": bool(c_scores and p_scores
                                                           and close_score(c_scores[0], p_scores[0])),
                                  "chroma_top": c_ids[:3], "pinecone_top": p_ids[:3],
                                  "chroma_top_score": c_scores[0] if c_scores else None,
                                  "pinecone_top_score": p_scores[0] if p_scores else None})
            if not per_query:
                return False, "no golden queries ran", per_query
            mean = sum(q["tie_adjusted"] for q in per_query) / len(per_query)
            worst = min(q["tie_adjusted"] for q in per_query)
            ok = mean >= args.min_mean_overlap and worst >= args.min_query_overlap
            return ok, (f"{len(per_query)} queries, mean overlap@{k} {mean:.3f} (min {worst:.3f}; thresholds "
                        f"{args.min_mean_overlap}/{args.min_query_overlap}); top-1 identical in "
                        f"{sum(q['top1_same'] for q in per_query)}; top-1 score consistent in "
                        f"{sum(q['top1_score_close'] for q in per_query)}"), per_query

        retry = not problems and now == expected_count
        while True:
            ok, text, per_query = run_golden()
            if ok or not retry or not queries or time.time() > search_deadline:
                break
            m.LOG.info("golden queries below threshold (%s); retrying in case of freshness lag", text)
            time.sleep(args.poll_interval)
        rep.add("golden_queries", "pass" if ok else "fail", text, queries=per_query[:50])

        hit1 = hitk = 0
        misses = []
        for sid in probe[: args.self_retrieval]:
            vec = src[sid]["embedding"]
            pid = cfg.id_map.get(sid, sid)
            res = search(vec, [cfg.vector_field])
            ids = [x.id for x in res.matches]
            hit1 += bool(ids and ids[0] == pid)
            hitk += pid in ids
            if not ids or ids[0] != pid:
                top = res.matches[0] if res.matches else None
                if top is not None and same_vector(vec, top.get(cfg.vector_field)):
                    why = "duplicate vector (tie with another record)"
                elif dense is not None and dense.metric == "dotproduct":
                    why = "dotproduct: an unnormalized vector is not always its own top match"
                elif pid in ids and close_score(res.matches[ids.index(pid)].score, top.score):
                    why = "near tie"
                elif pid not in ids:
                    why = "not in top-k: approximate search or freshness; rerun later"
                else:
                    why = "ranked below top-1: approximate search"
                misses.append({"id": sid[:80], "top1": ids[0][:80] if ids else None, "reason": why})
        n_probe = len(probe[: args.self_retrieval])
        rep.add("self_retrieval", "info",
                f"hit@1 {hit1}/{n_probe}, hit@{k} {hitk}/{n_probe} (diagnostic only)", misses=misses[:20])

    if args.app_filters:
        strf = {key for key, v in cfg.field_map.items()
                if cfg.array_policy == "stringify"
                and any(v.get("value_kinds", {}).get(x) for x in m.NON_STRING_ARRAY_KINDS)}
        arrays = array_keys_from_kinds({k: v.get("value_kinds", {}) for k, v in cfg.field_map.items()})
        entries = [(e, translate(e, cfg, strf, arrays)) for e in m.read_json(args.app_filters)]
        chroma_sets = {}
        for entry, t in entries:
            if t["pinecone_filter"] is None:
                continue
            c_ids, off = set(), 0
            while True:
                r = col.get(where=entry["where"], include=[], limit=args.page_size, offset=off)
                c_ids.update(cfg.id_map.get(x, x) for x in r["ids"])
                if len(r["ids"]) < args.page_size:
                    break
                off += args.page_size
            chroma_sets[t["name"]] = c_ids

        def pinecone_ids(flt):
            ids, token = set(), None
            while True:
                try:
                    r = index.documents.fetch(namespace=args.namespace, filter=flt,
                                              include_fields=[cfg.vector_field], limit=100,
                                              pagination_token=token)
                except NotFoundError:
                    return ids
                ids.update(r.documents.keys())
                if r.pagination is None:
                    return ids
                token = r.pagination.next

        filter_deadline = time.time() + args.search_timeout
        while True:
            rows, bad = [], 0
            for entry, t in entries:
                row = {"name": t["name"], "status": t["status"], "notes": t["notes"],
                       "pinecone_filter": t["pinecone_filter"]}
                if t["pinecone_filter"] is not None:
                    c_ids, p_ids = chroma_sets[t["name"]], pinecone_ids(t["pinecone_filter"])
                    row.update({"chroma_matches": len(c_ids), "pinecone_matches": len(p_ids),
                                "same_ids": c_ids == p_ids, "only_chroma": sorted(c_ids - p_ids)[:5],
                                "only_pinecone": sorted(p_ids - c_ids)[:5]})
                    bad += c_ids != p_ids
                rows.append(row)
            if not bad or not (not problems and now == expected_count) or time.time() > filter_deadline:
                break
            m.LOG.info("%d filters differ; retrying in case of freshness lag", bad)
            time.sleep(args.poll_interval)
        for row in rows:
            print(f"        filter {row['name']}: {row['status']} chroma={row.get('chroma_matches', '-')} "
                  f"pinecone={row.get('pinecone_matches', '-')} {'; '.join(row['notes'])}")
        redesign = [r["name"] for r in rows if r["status"] == "needs_redesign"]
        rep.add("app_filters", "fail" if bad else ("warn" if redesign else "pass"),
                f"{len(rows)} filters, {bad} with different results, {len(redesign)} need redesign {redesign}",
                filters=rows)

    out = args.report or m.wpath(args.workdir, m.VALIDATION_REPORT)
    m.write_json(out, {"index": args.index, "namespace": args.namespace, "collection": args.collection,
                       "validated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "ok": not rep.failed, "checks": rep.checks})
    print(f"report: {out}")
    if rep.failed:
        print("VALIDATION FAILED: stop, fix the cause, reload into a fresh namespace, validate again")
        return m.EXIT_FAIL
    print("VALIDATION OK")
    return m.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
