#!/usr/bin/env python3
"""Offline checks of the conversion rules, including inputs that local
chromadb 1.5.9 refuses to store but older or hosted data might contain
($-prefixed keys, None values, empty arrays, nested objects).

Run: python tests/test_convert_unit.py   (exit 0 = all passed)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import miglib as m  # noqa: E402
from filters import array_keys_from_kinds, translate  # noqa: E402

V = [0.1, 0.2, 0.3]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok  ", msg)


def cfg_for(records, **kw):
    cfg = m.ConvertConfig(**kw)
    an = m.Analyzer(cfg)
    for r in records:
        an.add(r)
    cfg.dimension = an.dimension()
    cfg.field_map = an.field_map()
    cfg.id_map = an.id_map()
    return cfg, an


def main():
    keys = ["text", "_text", "meta_text", "embedding", "$foo", "_id", "foo", "meta_foo", "k" * 70, "é" * 40,
            "chroma_id", "meta_embedding"]
    fmap = m.build_field_map(keys, {"embedding", "text", "chroma_id"})
    dests = [v["dest"] for v in fmap.values()]
    check(len(set(dests)) == len(dests), "field renames are collision-free")
    check(all(m.field_problem(d, {"embedding", "text", "chroma_id"}) is None for d in dests),
          "every destination name is valid")
    check(fmap["meta_text"]["dest"] == "meta_text" and fmap["foo"]["dest"] == "foo", "valid keys keep their name")
    check(fmap["$foo"]["dest"] != "meta_foo", "$foo does not take the existing meta_foo")
    check(fmap["embedding"]["dest"] != "meta_embedding", "embedding does not take the existing meta_embedding")
    check(m.build_field_map(list(reversed(keys)), {"embedding", "text", "chroma_id"}) == fmap,
          "rename table is independent of key order")

    collide = m.hashed_id("ü")
    idm = m.build_id_map(["ü", "x" * 513], {collide, "a"}, m.DEFAULT_ID_PREFIX)
    check(idm["ü"] == collide + "-2", "mapped ID that equals an existing ID gets a -2 suffix")
    check(all(m.id_problem(v) is None for v in idm.values()), "mapped IDs are valid")

    recs = [
        {"id": "a", "embedding": V, "document": "x", "metadata": {"$foo": 1, "n": None, "e": [], "s": ["p"]}},
        {"id": "b", "embedding": V, "document": None, "metadata": None},
        {"id": "c", "embedding": V, "document": "", "metadata": {"arr": [1, 2], "flags": [True, False],
                                                                   "f": [0.5], "big": 2 ** 60}},
        {"id": "d", "embedding": V, "document": "y", "metadata": {"nested": {"a": 1}}},
        {"id": "z", "embedding": [0.0, 0.0, 0.0], "document": "y", "metadata": None},
    ]
    cfg, an = cfg_for(recs)
    sev = {i["code"]: i["severity"] for i in an.issues()}
    check(sev.get("non_string_arrays") == "blocking", "numeric/bool arrays block under the default policy")
    check(sev.get("ints_beyond_2_53") == "blocking", "ints beyond 2^53 block under the default policy")
    check(sev.get("unsupported_value_types") == "blocking", "nested objects block")
    check(sev.get("zero_vector") == "blocking", "zero vectors block")
    check(sev.get("none_values") == "transform", "None values are a logged transform")
    check(sev.get("empty_arrays") == "info", "empty arrays are kept")

    doc, tr, err = m.convert_record(recs[2], cfg)
    check(doc is None and {e["code"] for e in err} == {"unsupported_array", "int_beyond_2_53"},
          "default policy rejects record c with explicit errors")

    cfg2, _ = cfg_for(recs[:3], array_policy="stringify", big_int_policy="stringify")
    out = {}
    for r in recs[:3]:
        doc, tr, err = m.convert_record(r, cfg2)
        check(not err, f"record {r['id']} converts under stringify policies")
        back = m.reverse_document(doc, cfg2, tr)
        check(back["id"] == r["id"] and back["document"] == r["document"], f"record {r['id']} id/document round-trip")
        check((back["metadata"] or {}) == (r["metadata"] or {}), f"record {r['id']} metadata round-trips: {back['metadata']}")
        out[r["id"]] = (doc, tr)
    doc_a, tr_a = out["a"]
    check(cfg2.field_map["$foo"]["dest"] in doc_a and "n" not in doc_a, "$foo renamed, None dropped")
    check(any(t["kind"] == "none_value_dropped" for t in tr_a), "None drop is logged")
    doc_b, tr_b = out["b"]
    check("text" not in doc_b and tr_b[0]["kind"] == "document_none_omitted", "None document omitted and logged")
    doc_c, _ = out["c"]
    check(doc_c["arr"] == ["1", "2"] and doc_c["flags"] == ["true", "false"] and doc_c["f"] == ["0.5"],
          "arrays stringified")
    check(doc_c["big"] == str(2 ** 60) and doc_c["text"] == "", "big int stringified, empty document kept")

    big = {"id": "big", "embedding": V, "document": "word " * 8200, "metadata": None}
    cfg3, an3 = cfg_for([big])
    check(any(i["code"] == "metadata_over_40kb" for i in an3.issues()), "41,000-char text (8,200 words) blocks in metadata mode")
    cfg4, an4 = cfg_for([big], text_mode="fts")
    check(not any(i["severity"] == "blocking" for i in an4.issues()), "the same text passes in fts mode (under 100 KB and 10,000 tokens)")
    check(m.convert_record(big, cfg4)[2] == [], "fts mode converts it")

    cfg5, an5 = cfg_for([{"id": "t", "embedding": V, "document": "x",
                          "metadata": {"tags": ["a"], "nums": [1], "text": "v", "cat": "a"}}],
                        array_policy="stringify")
    strf = {"nums"}
    arrays = array_keys_from_kinds({k: dict(c) for k, c in an5.key_kinds.items()})
    check(arrays == {"tags", "nums"}, "array keys come from observed value kinds")
    t = translate({"name": "c", "where": {"tags": {"$contains": "a"}}}, cfg5, strf, arrays)
    check(t["pinecone_filter"] == {"tags": {"$in": ["a"]}} and t["status"] == "translated_verify", "$contains -> $in")
    t = translate({"name": "n", "where": {"nums": {"$contains": 1}}}, cfg5, strf, arrays)
    check(t["pinecone_filter"] == {"nums": {"$in": ["1"]}}, "numeric $contains uses the stringified value")
    t = translate({"name": "r", "where": {"text": "v"}}, cfg5, strf, arrays)
    check(list(t["pinecone_filter"]) == [cfg5.field_map["text"]["dest"]], "filter on a renamed key is rewritten")
    t = translate({"name": "g", "where": {"nums": {"$gt": 1}}}, cfg5, strf, arrays)
    check(t["status"] == "needs_redesign", "range filter on a stringified array needs redesign")
    t = translate({"name": "d", "where_document": {"$contains": "x"}}, cfg5, strf, arrays)
    check(t["status"] == "needs_redesign", "where_document needs redesign")
    t = translate({"name": "s", "where": {"cat": {"$contains": "a"}}}, cfg5, strf, arrays)
    check(t["status"] == "needs_redesign" and t["pinecone_filter"] is None,
          "$contains on a scalar key is not rewritten to $in")
    t = translate({"name": "s2", "where": {"cat": {"$not_contains": "a"}}}, cfg5, strf, arrays)
    check(t["status"] == "needs_redesign", "$not_contains on a scalar key needs redesign")

    for text, ok in (("a " * 10000, True), ("a " * 10001, False), ("a." * 10001, False),
                     ("don't " * 6000, False), ("x-y-z " * 3500, False), ("a , " * 5001, True),
                     ("日本 " * 5001, True), ("日" * 9000, False), ("😀" * 10001, True)):
        n, longest = m.fts_token_stats(text)
        check((n <= m.FTS_FIELD_MAX_TOKENS and longest <= m.FTS_MAX_RAW_TOKEN_BYTES) == ok,
              f"token estimate {n} / longest {longest} bytes -> {'accepted' if ok else 'rejected'} "
              f"(matches server probe for {text[:8]!r}...)")
    tok = {"id": "tok", "embedding": V, "document": "a " * 30000, "metadata": None}
    cfg6, an6 = cfg_for([tok], text_mode="fts")
    check(any(i["code"] == "text_over_10000_tokens" and i["severity"] == "blocking" for i in an6.issues()),
          "30,000-word document blocks preflight in fts mode")
    check([e["code"] for e in m.convert_record(tok, cfg6)[2]] == ["document_text_over_10000_tokens"],
          "convert_record rejects it")
    cfg7, an7 = cfg_for([tok])
    check(any(i["code"] == "metadata_over_40kb" for i in an7.issues())
          and not any(i["code"] == "text_over_10000_tokens" for i in an7.issues()),
          "in metadata mode the 60 KB text is blocked by the 40 KB rule; token limits do not apply")
    cjk = {"id": "cjk", "embedding": V, "document": "日" * 400, "metadata": None}
    cfg8, an8 = cfg_for([cjk], text_mode="fts")
    check(any(i["code"] == "text_token_over_1024_bytes" for i in an8.issues()),
          "400 unspaced CJK characters (1,200 bytes, one token) block in fts mode")
    print("ALL UNIT CHECKS PASSED")


if __name__ == "__main__":
    main()
