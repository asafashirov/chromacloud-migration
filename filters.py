"""Translate Chroma `where` filters to Pinecone `filter` expressions.

Chroma and Pinecone share most Mongo-style operators. The differences that
matter for a migration:

* `$contains` / `$not_contains` in `where` test array membership. Pinecone has
  no such operator; on a list-of-strings field, `$in` with one value tests
  membership. `$not_contains` becomes `$nin`, which must be verified against
  your data (the filter parity check in validate.py does this). The rewrite is
  only made for keys whose exported values are all arrays: on a scalar key
  Chroma's `$contains` matches nothing while `$in` would match equal values.
* `where_document` (substring `$contains`, `$regex`) has no Pinecone
  equivalent. Full-text search operators ($match_phrase/$match_all/$match_any)
  match tokens, not substrings, and need a full_text_search field, which means
  a new index. These filters need redesign.
* Keys renamed by convert.py and values of stringified arrays must be
  rewritten with the mapping table.
"""

from __future__ import annotations

from typing import Any

from miglib import ConvertConfig, stringify_scalar

COMPARISON = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte"}
SET_OPS = {"$in", "$nin"}


class NeedsRedesign(Exception):
    pass


def array_keys_from_kinds(key_kinds: dict[str, dict]) -> set[str]:
    """Keys whose every observed value is an array."""
    out = set()
    for k, kinds in key_kinds.items():
        seen = [x for x in kinds if x != "none"]
        if seen and all(str(x).startswith("list_") for x in seen):
            out.add(k)
    return out


def rewrite_where(where: dict, cfg: ConvertConfig, stringified_keys: set[str], notes: list[str],
                  array_keys: set[str]) -> dict:
    if not isinstance(where, dict) or not where:
        raise NeedsRedesign(f"unsupported where clause: {where!r}")
    out: dict[str, Any] = {}
    for key, cond in where.items():
        if key in ("$and", "$or"):
            out[key] = [rewrite_where(c, cfg, stringified_keys, notes, array_keys) for c in cond]
            continue
        if key.startswith("$"):
            raise NeedsRedesign(f"unsupported top-level operator {key}")
        entry = cfg.field_map.get(key)
        dest = entry["dest"] if entry else key
        if entry is None:
            notes.append(f"key {key!r} does not occur in the exported metadata")
        elif dest != key:
            notes.append(f"key {key!r} renamed to {dest!r}")
        strf = key in stringified_keys

        def val(v):
            if strf and not isinstance(v, str):
                return stringify_scalar(v)
            return v

        if not isinstance(cond, dict):
            out[dest] = {"$eq": val(cond)}
            continue
        new: dict[str, Any] = {}
        for op, v in cond.items():
            if op in COMPARISON:
                if strf and op not in ("$eq", "$ne"):
                    raise NeedsRedesign(f"{op} on {key!r}, whose arrays were stringified")
                new[op] = val(v)
            elif op in SET_OPS:
                new[op] = [val(x) for x in v]
            elif op in ("$contains", "$not_contains") and key not in array_keys:
                raise NeedsRedesign(
                    f"{op} on {key!r}, which is not an array field in the export: Chroma's array "
                    "membership has no scalar equivalent here (rewriting to $in/$nin would change results)")
            elif op == "$contains":
                new["$in"] = [val(v)]
                notes.append(f"{key}: $contains rewritten as $in [value] (list membership)")
            elif op == "$not_contains":
                new["$nin"] = [val(v)]
                notes.append(f"{key}: $not_contains rewritten as $nin [value]; verify with the parity check")
            else:
                raise NeedsRedesign(f"operator {op} on {key!r} has no Pinecone equivalent")
        out[dest] = new
    return out


def translate(entry: dict, cfg: ConvertConfig, stringified_keys: set[str], array_keys: set[str]) -> dict:
    """entry: {"name": ..., "where": {...}} and/or {"where_document": {...}}."""
    result = {"name": entry.get("name"), "status": "translated", "notes": [], "pinecone_filter": None}
    if entry.get("where_document"):
        result["status"] = "needs_redesign"
        result["notes"].append(
            "where_document (substring/regex on document text) has no Pinecone equivalent. "
            "full_text_search operators match tokens rather than substrings and require a "
            "full_text_search field, which means creating a new index and reloading."
        )
        return result
    if not entry.get("where"):
        result["status"] = "needs_redesign"
        result["notes"].append("no where clause given")
        return result
    try:
        result["pinecone_filter"] = rewrite_where(entry["where"], cfg, stringified_keys, result["notes"],
                                                  array_keys)
    except NeedsRedesign as exc:
        result["status"] = "needs_redesign"
        result["notes"].append(str(exc))
    if result["status"] == "translated" and any("contains" in n for n in result["notes"]):
        result["status"] = "translated_verify"
    return result
