"""Shared logic for the Chroma Cloud -> Pinecone migration scripts.

The scripts move one Chroma collection that already has dense embeddings into a
Pinecone document index (a schema with one dense_vector field), keeping the
existing vectors, the document text and the metadata.

Every script imports this module, so preflight, convert and validate apply
exactly the same rules.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterator

LOG = logging.getLogger("chroma_migration")

PINECONE_ID_MAX_CHARS = 512
PINECONE_ID_RE = re.compile(r"^[\x01-\x7F]+$")
FIELD_NAME_MAX_BYTES = 64
METADATA_LIMIT_BYTES = 40960
FTS_FIELD_LIMIT_BYTES = 100_000
FTS_FIELD_MAX_TOKENS = 10_000
FTS_MAX_RAW_TOKEN_BYTES = 1024
FTS_TOKEN_RE = re.compile(r"[^\W_]+")
FTS_RAW_TOKEN_RE = re.compile(r"\w+")
MAX_DOCUMENT_BYTES = 2_000_000
MAX_REQUEST_BYTES = 2_000_000
MAX_DOCS_PER_UPSERT = 1000
MAX_IDS_PER_FETCH = 1000
MAX_SAFE_INT = 2 ** 53

METRIC_MAP = {"l2": "euclidean", "cosine": "cosine", "ip": "dotproduct"}
CHROMA_DEFAULT_SPACE = "l2"

DEFAULT_VECTOR_FIELD = "embedding"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_ID_FIELD = "chroma_id"
DEFAULT_ID_PREFIX = "chroma-sha256-"

PREFLIGHT_REPORT = "preflight.json"
EXPORT_FILE = "export.jsonl"
EXPORT_MANIFEST = "export_manifest.json"
DOCUMENTS_FILE = "documents.jsonl"
FIELD_MAP_FILE = "field_map.json"
ID_MAP_FILE = "id_map.jsonl"
TRANSFORMS_FILE = "transforms.jsonl"
CONVERT_SUMMARY = "convert_summary.json"
LOAD_STATE = "load_state.json"
VALIDATION_REPORT = "validation_report.json"

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    for noisy in ("httpx", "httpcore", "pinecone", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def utf8_len(s: str) -> int:
    return len(s.encode("utf-8", "surrogatepass"))


def dumps(obj: Any) -> str:
    """Compact JSON that refuses NaN/Infinity, so bad floats fail loudly."""
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def short(x: Any, n: int = 60) -> str:
    s = x if isinstance(x, str) else repr(x)
    return s if len(s) <= n else s[: n - 3] + "..."


def wpath(workdir: str, name: str) -> str:
    return os.path.join(workdir, name)


# ---------------------------------------------------------------------------
# Chroma side
# ---------------------------------------------------------------------------
def add_chroma_args(parser) -> None:
    parser.add_argument("--collection", required=True, help="Chroma collection name")
    parser.add_argument(
        "--chroma-path",
        help="Use a local chromadb.PersistentClient at this path (testing). "
        "Omit to use chromadb.CloudClient() with CHROMA_API_KEY, "
        "CHROMA_TENANT and CHROMA_DATABASE from the environment.",
    )


def get_chroma_collection(args):
    import chromadb

    if args.chroma_path:
        client = chromadb.PersistentClient(path=args.chroma_path)
    else:
        client = chromadb.CloudClient()
    return client.get_collection(args.collection, embedding_function=None)


def detect_metric(collection) -> dict:
    """Return the collection's distance space and where it was read from.

    Order: configuration_json (hnsw, then spann), the collection schema's
    #embedding vector index, the "hnsw:space" collection metadata key, and
    finally Chroma's default (l2).
    """
    sources: dict[str, str] = {}
    try:
        cfg = collection.configuration_json or {}
        for section in ("hnsw", "spann"):
            sec = cfg.get(section) or {}
            if isinstance(sec, dict) and sec.get("space"):
                sources[f"configuration_json.{section}.space"] = sec["space"]
    except Exception as exc:
        LOG.debug("configuration_json unavailable: %s", exc)
    try:
        schema = collection.schema
        if schema is not None:
            sj = schema.serialize_to_json()
            vi = (((sj.get("keys") or {}).get("#embedding") or {}).get("float_list") or {}).get(
                "vector_index"
            ) or {}
            space = (vi.get("config") or {}).get("space")
            if space:
                sources["schema.keys.#embedding.vector_index.config.space"] = space
    except Exception as exc:
        LOG.debug("schema unavailable: %s", exc)
    meta = collection.metadata or {}
    if isinstance(meta, dict) and meta.get("hnsw:space"):
        sources["metadata.hnsw:space"] = meta["hnsw:space"]

    if sources:
        source = next(iter(sources))
        space = sources[source]
    else:
        space, source = CHROMA_DEFAULT_SPACE, "chroma default (no space configured)"
    return {
        "space": space,
        "source": source,
        "all_sources": sources,
        "consistent": len(set(sources.values())) <= 1,
        "pinecone_metric": METRIC_MAP.get(space),
    }


def embedding_function_info(collection) -> Any:
    try:
        return (collection.configuration_json or {}).get("embedding_function")
    except Exception:
        return None


def _to_list(vec) -> list[float] | None:
    if vec is None:
        return None
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    return [float(x) for x in vec]


def iter_chroma_pages(collection, page_size: int, include: list[str]) -> Iterator[list[dict]]:
    """Stream a collection with limit/offset paging. Writes must be paused."""
    offset = 0
    while True:
        res = collection.get(include=include, limit=page_size, offset=offset)
        ids = res["ids"]
        if not ids:
            return
        embs = res.get("embeddings") if "embeddings" in include else None
        docs = res.get("documents") if "documents" in include else None
        metas = res.get("metadatas") if "metadatas" in include else None
        page = []
        for i, rid in enumerate(ids):
            rec: dict[str, Any] = {"id": rid}
            if "embeddings" in include:
                rec["embedding"] = _to_list(embs[i]) if embs is not None else None
            if "documents" in include:
                rec["document"] = docs[i] if docs is not None else None
            if "metadatas" in include:
                md = metas[i] if metas is not None else None
                rec["metadata"] = dict(sorted(md.items())) if md else None
            page.append(rec)
        yield page
        if len(ids) < page_size:
            return
        offset += len(ids)


def chroma_get_records(collection, ids: list[str]) -> dict[str, dict]:
    out = {}
    for start in range(0, len(ids), 100):
        res = collection.get(ids=ids[start : start + 100], include=["embeddings", "documents", "metadatas"])
        embs = res.get("embeddings")
        for i, rid in enumerate(res["ids"]):
            md = res["metadatas"][i]
            out[rid] = {
                "id": rid,
                "embedding": _to_list(embs[i]) if embs is not None else None,
                "document": res["documents"][i],
                "metadata": dict(sorted(md.items())) if md else None,
            }
    return out


# ---------------------------------------------------------------------------
# Pinecone side
# ---------------------------------------------------------------------------
def get_pinecone():
    from pinecone import Pinecone

    if not os.environ.get("PINECONE_API_KEY"):
        LOG.error("PINECONE_API_KEY is not set")
        raise SystemExit(EXIT_USAGE)
    return Pinecone()


def open_pinecone_index(pc, name: str):
    """Return (index_model, index client); exit 2 if the index does not exist."""
    from pinecone.errors.exceptions import NotFoundError

    try:
        return pc.indexes.describe(name=name), pc.index(name=name)
    except NotFoundError:
        LOG.error("Pinecone index %r does not exist in this project", name)
        raise SystemExit(EXIT_USAGE)


def schema_fields(index_model) -> dict:
    return dict(getattr(index_model.schema, "fields", None) or {})


# ---------------------------------------------------------------------------
# IDs
# ---------------------------------------------------------------------------
def id_problem(rid: Any) -> str | None:
    if not isinstance(rid, str):
        return "not_a_string"
    if rid == "":
        return "empty"
    if len(rid) > PINECONE_ID_MAX_CHARS:
        return "longer_than_512_chars"
    if not PINECONE_ID_RE.fullmatch(rid):
        return "non_ascii_or_nul"
    return None


def hashed_id(rid: str, prefix: str = DEFAULT_ID_PREFIX) -> str:
    return prefix + hashlib.sha256(rid.encode("utf-8", "surrogatepass")).hexdigest()


def build_id_map(invalid_ids: list[str], valid_ids: set[str], prefix: str) -> dict[str, str]:
    """Deterministic, collision-free mapping for IDs Pinecone rejects.

    new_id = prefix + sha256(utf8(source_id)).hex. If that string is already a
    valid source ID or was already assigned, append -2, -3, ... Invalid IDs are
    processed in sorted order, so the result does not depend on export order.
    """
    taken = set(valid_ids)
    out: dict[str, str] = {}
    for rid in sorted(set(invalid_ids)):
        base = hashed_id(rid, prefix)
        cand, n = base, 1
        while cand in taken:
            n += 1
            cand = f"{base}-{n}"
        taken.add(cand)
        out[rid] = cand
    return out


# ---------------------------------------------------------------------------
# field names
# ---------------------------------------------------------------------------
def field_problem(name: str, reserved: set[str]) -> str | None:
    if not isinstance(name, str) or name == "":
        return "empty"
    try:
        n = len(name.encode("utf-8"))
    except UnicodeEncodeError:
        return "invalid_unicode"
    if name.startswith("_"):
        return "starts_with_underscore"
    if name.startswith("$"):
        return "starts_with_dollar"
    if n > FIELD_NAME_MAX_BYTES:
        return "longer_than_64_bytes"
    if name in reserved:
        return "reserved_name"
    return None


def _truncate_utf8(s: str, max_bytes: int) -> str:
    return s.encode("utf-8", "surrogatepass")[:max_bytes].decode("utf-8", "ignore")


def build_field_map(keys: list[str], reserved: set[str]) -> dict[str, dict]:
    """Collision-free, reversible renaming table: {source_key: {dest, reason}}.

    Valid keys keep their name. An invalid key k becomes "meta_" + k with
    leading "_"/"$" stripped. If that name is taken (by a kept key, a reserved
    field or an earlier rename) or still invalid, it becomes
    "meta_<truncated>_<sha256(k)[:8]>", then "-2", "-3", ... Invalid keys are
    processed in sorted order, so the table is deterministic.
    """
    keys = sorted(set(keys))
    table: dict[str, dict] = {}
    taken = set(reserved)
    invalid = []
    for k in keys:
        reason = field_problem(k, reserved)
        if reason is None:
            table[k] = {"dest": k, "reason": None}
            taken.add(k)
        else:
            invalid.append((k, reason))
    for k, reason in invalid:
        stripped = k.lstrip("_$") or "key"
        cand = "meta_" + stripped
        if cand in taken or field_problem(cand, reserved) is not None:
            h = hashlib.sha256(k.encode("utf-8", "surrogatepass")).hexdigest()[:8]
            cand = _truncate_utf8("meta_" + stripped, FIELD_NAME_MAX_BYTES - 12) + "_" + h
            base, n = cand, 1
            while cand in taken:
                n += 1
                cand = f"{base}-{n}"
        if field_problem(cand, reserved) is not None:
            raise ValueError(f"could not build a valid name for key {k!r}")
        taken.add(cand)
        table[k] = {"dest": cand, "reason": reason}
    return table


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------
def value_kind(v: Any) -> str:
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return "list_empty"
        kinds = {value_kind(x) for x in v}
        if len(kinds) == 1:
            k = next(iter(kinds))
            if k in ("str", "int", "float", "bool"):
                return "list_" + k
        if kinds <= {"int", "float"}:
            return "list_number"
        return "list_mixed"
    return "other:" + type(v).__name__


NON_STRING_ARRAY_KINDS = ("list_int", "list_float", "list_bool", "list_number", "list_mixed")


def stringify_scalar(x: Any) -> str:
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, float):
        return repr(x)
    return str(x)


def fts_token_stats(text: str) -> tuple[int, int]:
    """Estimate (token count, largest raw token in bytes) for a full_text_search field.

    Pinecone splits on whitespace and punctuation (hyphens and apostrophes
    included). Counting runs of letters/digits, with "_" as a separator, matched
    the server on every probe we ran (10,000 tokens accepted, 10,001 rejected).
    The server also rejects any single token over 1,024 UTF-8 bytes (its error
    message states that limit); the raw-token size uses \\w+ runs, the longer
    reading, so both numbers err on the high side.
    """
    count = sum(1 for _ in FTS_TOKEN_RE.finditer(text))
    longest = max((utf8_len(t.group()) for t in FTS_RAW_TOKEN_RE.finditer(text)), default=0)
    return count, longest


def unstringify_scalar(s: str, kind: str) -> Any:
    if kind == "bool":
        return s == "true"
    if kind == "int":
        return int(s)
    if kind == "float":
        return float(s)
    return s


def metadata_size_estimate(fields: dict[str, Any]) -> int:
    """Approximate Pinecone's metadata accounting as key bytes + value bytes.

    This matches the server for string values (a 41,000-character value under
    key "big" was reported as 41003 bytes). Numbers count 8, booleans 1 and
    string lists the sum of their element bytes. The server stays the authority.
    """
    total = 0
    for k, v in fields.items():
        total += utf8_len(k)
        if isinstance(v, str):
            total += utf8_len(v)
        elif isinstance(v, bool):
            total += 1
        elif isinstance(v, (int, float)):
            total += 8
        elif isinstance(v, list):
            total += sum(utf8_len(x) if isinstance(x, str) else 8 for x in v)
    return total


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------
@dataclass
class ConvertConfig:
    vector_field: str = DEFAULT_VECTOR_FIELD
    text_field: str = DEFAULT_TEXT_FIELD
    id_field: str | None = DEFAULT_ID_FIELD
    id_prefix: str = DEFAULT_ID_PREFIX
    text_mode: str = "metadata"
    array_policy: str = "reject"
    big_int_policy: str = "reject"
    dimension: int | None = None
    field_map: dict[str, dict] = field(default_factory=dict)
    id_map: dict[str, str] = field(default_factory=dict)

    @property
    def reserved(self) -> set[str]:
        r = {self.vector_field, self.text_field}
        if self.id_field:
            r.add(self.id_field)
        return r

    def to_json(self) -> dict:
        return {
            "vector_field": self.vector_field,
            "text_field": self.text_field,
            "id_field": self.id_field,
            "id_prefix": self.id_prefix,
            "text_mode": self.text_mode,
            "array_policy": self.array_policy,
            "big_int_policy": self.big_int_policy,
            "dimension": self.dimension,
        }


def add_convert_policy_args(parser) -> None:
    parser.add_argument("--array-policy", choices=["reject", "stringify"], default="reject",
                        help="Numeric/boolean metadata arrays: reject (default) or convert "
                        "elements to strings (3 -> '3', True -> 'true').")
    parser.add_argument("--big-int-policy", choices=["reject", "stringify"], default="reject",
                        help="Integers beyond 2^53: reject (default) or store as strings.")
    parser.add_argument("--text-mode", choices=["metadata", "fts"], default="metadata",
                        help="metadata (default): document text is a metadata field and "
                        "counts toward the 40 KB limit. fts: the index declares the text field "
                        "with full_text_search (create_index.py --full-text-search).")
    parser.add_argument("--vector-field", default=DEFAULT_VECTOR_FIELD)
    parser.add_argument("--text-field", default=DEFAULT_TEXT_FIELD)
    parser.add_argument("--id-field", default=DEFAULT_ID_FIELD,
                        help="Field that stores the original Chroma ID on remapped records "
                        "('' to disable).")
    parser.add_argument("--id-prefix", default=DEFAULT_ID_PREFIX)


def config_from_args(args) -> ConvertConfig:
    return ConvertConfig(
        vector_field=args.vector_field,
        text_field=args.text_field,
        id_field=args.id_field or None,
        id_prefix=args.id_prefix,
        text_mode=args.text_mode,
        array_policy=args.array_policy,
        big_int_policy=args.big_int_policy,
    )


def load_config(workdir: str) -> tuple[ConvertConfig, dict]:
    fm = read_json(wpath(workdir, FIELD_MAP_FILE))
    c = fm["config"]
    cfg = ConvertConfig(
        vector_field=c["vector_field"],
        text_field=c["text_field"],
        id_field=c["id_field"],
        id_prefix=c["id_prefix"],
        text_mode=c["text_mode"],
        array_policy=c["array_policy"],
        big_int_policy=c["big_int_policy"],
        dimension=c["dimension"],
        field_map=fm["fields"],
    )
    idp = wpath(workdir, ID_MAP_FILE)
    if os.path.exists(idp):
        for row in iter_jsonl(idp):
            cfg.id_map[row["source_id"]] = row["pinecone_id"]
    return cfg, fm


def check_embedding(emb: Any, dimension: int | None) -> str | None:
    if emb is None:
        return "missing_embedding"
    if not isinstance(emb, list) or len(emb) == 0:
        return "empty_embedding"
    if dimension is not None and len(emb) != dimension:
        return "dimension_mismatch"
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in emb):
        return "non_finite_embedding"
    if all(x == 0 for x in emb):
        return "zero_vector"
    return None


def convert_record(rec: dict, cfg: ConvertConfig) -> tuple[dict | None, list[dict], list[dict]]:
    """Convert one exported Chroma record into a Pinecone document.

    Returns (document or None, transforms, errors). Each renamed-ID, rewritten
    or dropped value yields a transform entry. Anything Pinecone cannot store
    yields an error and no document.
    """
    sid = rec["id"]
    pid = cfg.id_map.get(sid, sid)
    transforms: list[dict] = []
    errors: list[dict] = []

    def tr(kind, **kw):
        transforms.append({"source_id": sid, "pinecone_id": pid, "kind": kind, **kw})

    def err(code, **kw):
        errors.append({"source_id": sid, "code": code, **kw})

    if id_problem(pid):
        err("invalid_id_not_mapped", reason=id_problem(pid))
    emb_problem = check_embedding(rec.get("embedding"), cfg.dimension)
    if emb_problem:
        err(emb_problem)

    doc: dict[str, Any] = {"_id": pid, cfg.vector_field: rec.get("embedding")}
    if pid != sid:
        tr("id_remapped", reason=id_problem(sid))
        if cfg.id_field:
            doc[cfg.id_field] = sid

    text = rec.get("document")
    if text is None:
        tr("document_none_omitted", field=cfg.text_field)
    elif not isinstance(text, str):
        err("document_not_a_string")
    elif text == "" and cfg.text_mode == "fts":
        tr("document_empty_omitted", field=cfg.text_field)
    else:
        doc[cfg.text_field] = text
        if cfg.text_mode == "fts":
            if utf8_len(text) > FTS_FIELD_LIMIT_BYTES:
                err("document_text_over_100kb", bytes=utf8_len(text))
            tokens, longest = fts_token_stats(text)
            if tokens > FTS_FIELD_MAX_TOKENS:
                err("document_text_over_10000_tokens", estimated_tokens=tokens,
                    hint="split the text into smaller documents in the source")
            if longest > FTS_MAX_RAW_TOKEN_BYTES:
                err("document_token_over_1024_bytes", longest_token_bytes=longest,
                    hint="text without spaces (for example CJK) forms one token; split or space it")

    for key, value in (rec.get("metadata") or {}).items():
        entry = cfg.field_map.get(key)
        if entry is None:
            err("key_not_in_field_map", key=key)
            continue
        dest = entry["dest"]
        kind = value_kind(value)
        if kind == "none":
            tr("none_value_dropped", key=key, dest=dest)
        elif kind in ("str", "bool", "list_str", "list_empty"):
            doc[dest] = list(value) if isinstance(value, (list, tuple)) else value
        elif kind == "float":
            if math.isfinite(value):
                doc[dest] = value
            else:
                err("non_finite_float", key=key)
        elif kind == "int":
            if abs(value) <= MAX_SAFE_INT:
                doc[dest] = value
            elif cfg.big_int_policy == "stringify":
                doc[dest] = str(value)
                tr("big_int_stringified", key=key, dest=dest)
            else:
                err("int_beyond_2_53", key=key)
        elif kind in NON_STRING_ARRAY_KINDS:
            elem_kinds = [value_kind(x) for x in value]
            if cfg.array_policy != "stringify":
                err("unsupported_array", key=key, kind=kind)
            elif any(k not in ("str", "int", "float", "bool") for k in elem_kinds):
                err("unsupported_array_element", key=key, kind=kind)
            else:
                doc[dest] = [stringify_scalar(x) for x in value]
                tr("array_stringified", key=key, dest=dest, element_kinds=elem_kinds)
        else:
            err("unsupported_value_type", key=key, kind=kind)

    meta_fields = {
        k: v
        for k, v in doc.items()
        if k not in ("_id", cfg.vector_field) and not (cfg.text_mode == "fts" and k == cfg.text_field)
    }
    msize = metadata_size_estimate(meta_fields)
    if msize > METADATA_LIMIT_BYTES:
        err("metadata_over_40kb", estimated_bytes=msize, text_counts=cfg.text_mode == "metadata")
    if not errors:
        try:
            dbytes = len(dumps(doc).encode("utf-8"))
        except ValueError as exc:
            err("not_json_serializable", detail=str(exc))
        else:
            if dbytes > MAX_DOCUMENT_BYTES:
                err("document_over_2mb", bytes=dbytes)
    return (None if errors else doc), transforms, errors


def reverse_document(doc: dict, cfg: ConvertConfig, record_transforms: list[dict]) -> dict:
    """Invert convert_record with the mapping files, for round-trip checks."""
    rev_fields = {v["dest"]: k for k, v in cfg.field_map.items()}
    rev_ids = {v: k for k, v in cfg.id_map.items()}
    pid = doc["_id"]
    sid = rev_ids.get(pid, pid)
    by_key: dict[str, list] = defaultdict(list)
    for t in record_transforms:
        if "key" in t:
            by_key[t["key"]].append(t)
    meta: dict[str, Any] = {}
    for k, v in doc.items():
        if k in ("_id", "_score", cfg.vector_field, cfg.text_field):
            continue
        if cfg.id_field and k == cfg.id_field and sid != pid:
            continue
        src = rev_fields.get(k, k)
        for t in by_key.get(src, []):
            if t["kind"] == "array_stringified":
                v = [unstringify_scalar(s, kd) for s, kd in zip(v, t["element_kinds"])]
            elif t["kind"] == "big_int_stringified":
                v = int(v)
        meta[src] = v
    for src, ts in by_key.items():
        if any(t["kind"] == "none_value_dropped" for t in ts):
            meta[src] = None
    document = doc.get(cfg.text_field)
    if document is None and any(t["kind"] == "document_empty_omitted" for t in record_transforms):
        document = ""
    return {
        "id": sid,
        "embedding": doc.get(cfg.vector_field),
        "document": document,
        "metadata": meta or None,
    }


# ---------------------------------------------------------------------------
# analysis shared by preflight and convert
# ---------------------------------------------------------------------------
class Analyzer:
    """Streaming scan that collects what preflight and convert need.

    Holds every ID in memory (needed for duplicate and collision checks);
    everything else is counters.
    """

    def __init__(self, cfg: ConvertConfig, example_cap: int = 5):
        self.cfg = cfg
        self.cap = example_cap
        self.count = 0
        self.dims: Counter = Counter()
        self.emb_problems: dict[str, list] = defaultdict(list)
        self.emb_problem_counts: Counter = Counter()
        self.valid_ids: set[str] = set()
        self.invalid_ids: list[str] = []
        self.invalid_id_reasons: Counter = Counter()
        self.duplicate_ids: list[str] = []
        self._seen_invalid: set[str] = set()
        self.key_kinds: dict[str, Counter] = defaultdict(Counter)
        self.docs_none = 0
        self.docs_empty = 0
        self.docs_non_ascii = 0
        self.max_doc_bytes = 0
        self.metadata_none = 0
        self.norm_min = math.inf
        self.norm_max = 0.0
        self.big_ints: Counter = Counter()
        self.nonfinite_floats: Counter = Counter()
        self.oversize: list[tuple[str, int]] = []
        self.near_limit = 0
        self.fts_over: list[tuple[str, int]] = []
        self.fts_over_tokens: list[tuple[str, int]] = []
        self.fts_long_tokens: list[tuple[str, int]] = []
        self.fts_token_warn = 0

    def add(self, rec: dict) -> None:
        self.count += 1
        rid = rec["id"]
        p = id_problem(rid)
        if p is None:
            if rid in self.valid_ids:
                self.duplicate_ids.append(rid)
            self.valid_ids.add(rid)
        else:
            if rid in self._seen_invalid:
                self.duplicate_ids.append(rid)
            else:
                self.invalid_ids.append(rid)
                self.invalid_id_reasons[p] += 1
            self._seen_invalid.add(rid)
        emb = rec.get("embedding")
        if isinstance(emb, list):
            self.dims[len(emb)] += 1
        ep = check_embedding(emb, None)
        if ep:
            self.emb_problem_counts[ep] += 1
            if len(self.emb_problems[ep]) < self.cap:
                self.emb_problems[ep].append(rid)
        else:
            n = math.sqrt(sum(x * x for x in emb))
            self.norm_min = min(self.norm_min, n)
            self.norm_max = max(self.norm_max, n)
        doc = rec.get("document")
        if doc is None:
            self.docs_none += 1
        elif doc == "":
            self.docs_empty += 1
        elif isinstance(doc, str):
            b = utf8_len(doc)
            self.max_doc_bytes = max(self.max_doc_bytes, b)
            if b != len(doc):
                self.docs_non_ascii += 1
            if self.cfg.text_mode == "fts":
                if b > FTS_FIELD_LIMIT_BYTES:
                    self.fts_over.append((rid, b))
                tokens, longest = fts_token_stats(doc)
                if tokens > FTS_FIELD_MAX_TOKENS:
                    self.fts_over_tokens.append((rid, tokens))
                elif tokens > FTS_FIELD_MAX_TOKENS * 0.8:
                    self.fts_token_warn += 1
                if longest > FTS_MAX_RAW_TOKEN_BYTES:
                    self.fts_long_tokens.append((rid, longest))
        meta = rec.get("metadata")
        if not meta:
            self.metadata_none += 1
            meta = {}
        for k, v in meta.items():
            kind = value_kind(v)
            self.key_kinds[k][kind] += 1
            if kind == "int" and abs(v) > MAX_SAFE_INT:
                self.big_ints[k] += 1
            if kind == "float" and not math.isfinite(v):
                self.nonfinite_floats[k] += 1
        fields = {
            k: ([stringify_scalar(x) for x in v] if isinstance(v, list) else v)
            for k, v in meta.items()
            if v is not None
        }
        if isinstance(doc, str) and self.cfg.text_mode == "metadata":
            fields[self.cfg.text_field] = doc
        est = metadata_size_estimate(fields)
        if est > METADATA_LIMIT_BYTES:
            self.oversize.append((rid, est))
        elif est > 0.9 * METADATA_LIMIT_BYTES:
            self.near_limit += 1

    def field_map(self) -> dict[str, dict]:
        return build_field_map(list(self.key_kinds), self.cfg.reserved)

    def id_map(self) -> dict[str, str]:
        return build_id_map(self.invalid_ids, self.valid_ids, self.cfg.id_prefix)

    def unique_ids(self) -> int:
        return len(self.valid_ids) + len(self._seen_invalid)

    def dimension(self) -> int | None:
        return self.dims.most_common(1)[0][0] if self.dims else None

    def key_report(self) -> dict:
        fmap = self.field_map()
        return {
            k: {"value_kinds": dict(c), "pinecone_field": fmap[k]["dest"], "rename_reason": fmap[k]["reason"]}
            for k, c in sorted(self.key_kinds.items())
        }

    def issues(self, metric: dict | None = None) -> list[dict]:
        """Severity levels: blocking (convert.py cannot fix it), transform
        (convert.py rewrites it and logs every change), warning, info."""
        out: list[dict] = []

        def add(code, severity, message, count=None, examples=None):
            out.append({"code": code, "severity": severity, "message": message,
                        "count": count, "examples": list(examples or [])[: self.cap]})

        if self.count == 0:
            add("empty_collection", "warning",
                "Collection has 0 records. The dimension cannot be inferred, so pass "
                "--dimension to create_index.py if you still want an index.")
        if self.duplicate_ids:
            add("duplicate_ids_in_scan", "blocking",
                "An ID appeared twice while paging. Writes are probably not paused.",
                len(self.duplicate_ids), self.duplicate_ids)
        if len(self.dims) > 1:
            add("inconsistent_dimension", "blocking",
                f"Embeddings have several dimensions: {dict(self.dims)}", sum(self.dims.values()))
        emb_msgs = {
            "zero_vector": "All-zero vectors are rejected by Pinecone (400). Fix or remove these "
                           "records in the source first.",
            "missing_embedding": "Record has no embedding; the migration needs one per record.",
            "empty_embedding": "Record has an empty embedding.",
            "non_finite_embedding": "Embedding contains NaN or Infinity.",
        }
        for code, n in self.emb_problem_counts.items():
            add(code, "blocking", emb_msgs.get(code, code), n, self.emb_problems[code])
        if self.invalid_ids:
            keep = (f" The original ID is stored in field '{self.cfg.id_field}'." if self.cfg.id_field else "")
            add("invalid_ids", "transform",
                f"IDs Pinecone rejects ({dict(self.invalid_id_reasons)}). convert.py maps each one to "
                f"'{self.cfg.id_prefix}<sha256 of the ID>' and records it in {ID_MAP_FILE}.{keep}",
                len(self.invalid_ids), self.invalid_ids)
        fmap = self.field_map()
        renamed = {k: v for k, v in fmap.items() if v["reason"]}
        if renamed:
            add("renamed_keys", "transform",
                f"Keys that are reserved or invalid in Pinecone are renamed (table in {FIELD_MAP_FILE}): "
                + ", ".join(f"{k!r} -> {v['dest']!r} ({v['reason']})" for k, v in renamed.items()),
                len(renamed), list(renamed))
        arrays = {k: {x: c[x] for x in NON_STRING_ARRAY_KINDS if c.get(x)} for k, c in self.key_kinds.items()}
        arrays = {k: v for k, v in arrays.items() if v}
        if arrays:
            sev = "transform" if self.cfg.array_policy == "stringify" else "blocking"
            tail = ("convert.py stringifies the elements (3 -> '3', True -> 'true'); numeric range "
                    "filters on these keys stop working." if sev == "transform" else
                    "Pass --array-policy stringify to convert them, or change the source data.")
            add("non_string_arrays", sev,
                f"Pinecone metadata lists must be lists of strings. Keys with other arrays: {arrays}. " + tail,
                sum(sum(v.values()) for v in arrays.values()), list(arrays))
        others = {k: [x for x in c if x.startswith("other:")] for k, c in self.key_kinds.items()}
        others = {k: v for k, v in others.items() if v}
        if others:
            add("unsupported_value_types", "blocking", f"Values Pinecone cannot store: {others}",
                len(others), list(others))
        nones = {k: c["none"] for k, c in self.key_kinds.items() if c.get("none")}
        if nones:
            add("none_values", "transform",
                f"None values are dropped (Pinecone metadata has no null) and every drop is logged: {nones}",
                sum(nones.values()), list(nones))
        empties = {k: c["list_empty"] for k, c in self.key_kinds.items() if c.get("list_empty")}
        if empties:
            add("empty_arrays", "info", f"Empty arrays are kept as [] (Pinecone accepts them): {empties}",
                sum(empties.values()), list(empties))
        if self.big_ints:
            sev = "transform" if self.cfg.big_int_policy == "stringify" else "blocking"
            add("ints_beyond_2_53", sev,
                f"Integers beyond 2^53 lose precision as 64-bit floats: {dict(self.big_ints)}. "
                + ("They are stored as strings." if sev == "transform" else "Pass --big-int-policy stringify."),
                sum(self.big_ints.values()), list(self.big_ints))
        if self.nonfinite_floats:
            add("non_finite_floats", "blocking", f"NaN or Infinity metadata: {dict(self.nonfinite_floats)}",
                sum(self.nonfinite_floats.values()), list(self.nonfinite_floats))
        ints = [k for k, c in self.key_kinds.items() if c.get("int")]
        if ints:
            add("ints_become_floats", "info",
                "Pinecone stores numbers as 64-bit floats, so ints read back as floats (1 -> 1.0). "
                "Numeric filters still match.", len(ints), ints)
        if self.oversize:
            add("metadata_over_40kb", "blocking",
                "Estimated metadata exceeds 40,960 bytes"
                + (" (document text counts, because text_mode=metadata)" if self.cfg.text_mode == "metadata" else "")
                + ". Shorten or split those records in the source, or create the index with "
                "--full-text-search and convert with --text-mode fts (text up to 100 KB).",
                len(self.oversize), [f"{i} (~{b} bytes)" for i, b in self.oversize])
        if self.near_limit:
            add("metadata_near_40kb", "warning", "Records within 10% of the 40 KB metadata limit.",
                self.near_limit)
        if self.fts_over:
            add("text_over_100kb", "blocking",
                "Document text over 100 KB does not fit a full_text_search field.",
                len(self.fts_over), [i for i, _ in self.fts_over])
        if self.fts_over_tokens:
            add("text_over_10000_tokens", "blocking",
                "Estimated full_text_search tokens exceed Pinecone's 10,000-token per-field limit (the "
                "upsert would fail with a 400 during the write pause). Split these documents into smaller "
                "ones in the source, or keep text_mode=metadata if they fit in 40 KB.",
                len(self.fts_over_tokens), [f"{i} (~{n} tokens)" for i, n in self.fts_over_tokens])
        if self.fts_long_tokens:
            add("text_token_over_1024_bytes", "blocking",
                "A single token (a run of letters/digits with no space or punctuation, such as unspaced "
                "CJK text) exceeds the 1,024-byte per-token limit of full_text_search fields.",
                len(self.fts_long_tokens), [f"{i} ({b} bytes)" for i, b in self.fts_long_tokens])
        if self.fts_token_warn:
            add("text_near_10k_tokens", "warning",
                "Documents with more than 8,000 estimated tokens (limit 10,000 per full_text_search field).",
                self.fts_token_warn)
        if self.docs_none:
            add("documents_none", "info",
                f"Records without a document: the '{self.cfg.text_field}' field is omitted (logged).",
                self.docs_none)
        if self.docs_empty:
            if self.cfg.text_mode == "fts":
                add("documents_empty", "info", "Empty-string documents: a full_text_search field does not "
                    "store '', so the field is omitted (logged).", self.docs_empty)
            else:
                add("documents_empty", "info", "Empty-string documents are kept as ''.", self.docs_empty)
        if metric:
            if not metric["consistent"]:
                add("metric_sources_disagree", "blocking",
                    f"The collection reports different spaces: {metric['all_sources']}")
            if metric.get("pinecone_metric") is None:
                add("unknown_metric", "blocking", f"Unknown Chroma space {metric['space']!r}")
            if metric["space"] == "ip" and self.norm_max > 0 and (
                abs(self.norm_max - 1) > 1e-3 or abs(self.norm_min - 1) > 1e-3
            ):
                add("ip_unnormalized", "info",
                    f"ip collection with unnormalized vectors (norms {self.norm_min:.3f} to "
                    f"{self.norm_max:.3f}). dotproduct ranks the same way, but a record is not "
                    "always its own nearest neighbor, so self-retrieval misses are expected.")
        return out
