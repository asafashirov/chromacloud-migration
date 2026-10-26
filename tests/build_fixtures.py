#!/usr/bin/env python3
"""Build the QA edge-case collections in a local Chroma (PersistentClient).

Collections (all 8-dimensional):
  qa_main     l2 (Chroma default), 251 records: invalid IDs, an ID that collides
              with a mapped ID, key collisions (text/_text/meta_text/embedding/
              chroma_id, >64-byte keys), int/float/bool/str values, string,
              numeric and boolean arrays, None and empty documents, non-ASCII
              text, duplicate vectors, ~30 KB documents (request-size stress),
              one ~39.5 KB document (near the 40 KB metadata limit)
  qa_cosine   cosine, 60 records
  qa_ip       ip, 60 unnormalized records
  qa_empty    l2, 0 records
  qa_blocked  l2: a zero vector and a 45,000-character document (both blocking)
  qa_fts      l2: a 60,000-character document (only loads with --text-mode fts)
  qa_tok_edge l2: exactly 10,000 full-text tokens and a 1,024-byte token (both allowed)
  qa_tok_over l2: 30,000 tokens and a 1,025-byte token (both blocked in fts mode)
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import chromadb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import miglib as m  # noqa: E402

DIM = 8
WORDS = "vector search index namespace cosine euclidean migration chroma pinecone record field".split()


def vec(rng: random.Random, scale: float = 1.0) -> list[float]:
    return [rng.gauss(0, 1) * scale for _ in range(DIM)]


def unit(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def text(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def fresh(client, name, **kw):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name, embedding_function=None, **kw)


def build_main(client, rng):
    col = fresh(client, "qa_main")
    ids, embs, docs, metas = [], [], [], []
    cats = ["news", "blog", "docs", "forum", "faq"]
    tags = ["red", "green", "blue", "café", "日本"]
    for i in range(240):
        ids.append(f"doc-{i:04d}")
        embs.append(vec(rng, rng.uniform(0.5, 3.0)))
        if 100 <= i < 200:
            docs.append(text(rng, 3300)[:30000])
        elif i % 17 == 0:
            docs.append("Ünïcödé text: " + text(rng, 12) + " 東京")
        else:
            docs.append("The " + text(rng, 20))
        md = {"category": cats[i % 5], "year": 2010 + i % 15, "score": round(rng.random(), 4),
              "published": i % 2 == 0}
        if i % 3:
            md["tags"] = rng.sample(tags, 1 + i % 3)
        if i % 10 == 0:
            md["ratings"] = [i % 5, (i + 2) % 5 + 1]
        if i % 15 == 0:
            md["bools"] = [True, i % 2 == 0]
        if i % 7 == 0:
            md["weights"] = [0.5, 1.25]
        metas.append(md)

    collide_target = "café-☕"
    extras = [
        (collide_target, vec(rng), "non-ASCII id", {"category": "news"}),
        ("x" * 600, vec(rng), "600-character id", {"category": "blog"}),
        (m.hashed_id(collide_target), vec(rng), "valid id equal to the mapped form of café-☕", {"category": "docs"}),
        ("naïve-id", vec(rng), "another non-ASCII id", None),
        ("keys-1", vec(rng), "key collision record", {
            "text": "meta text value", "_text": "underscore text", "meta_text": "explicit meta_text",
            "embedding": "not a vector", "chroma_id": "user field named like the id field",
            "k" * 70: 1, "é" * 40: "80-byte key"}),
        ("doc-nodoc", vec(rng), None, {"category": "faq", "year": 2020}),
        ("doc-emptydoc", vec(rng), "", {"category": "faq", "score": 1.0}),
        ("dup-a", [1.0, 2.0, 3.0, 4.0, 0.5, 0.25, 0.125, 1.0], "duplicate vector A", {"category": "forum"}),
        ("dup-b", [1.0, 2.0, 3.0, 4.0, 0.5, 0.25, 0.125, 1.0], "duplicate vector B", {"category": "forum"}),
        ("near-40kb", vec(rng), "n" * 39500, {"category": "docs"}),
        ("doc-extra", vec(rng), "extra record so the count is 250 after one delete", {"year": 1999}),
    ]
    for rid, e, d, md in extras:
        ids.append(rid)
        embs.append(e)
        docs.append(d)
        metas.append(md)
    for start in range(0, len(ids), 50):
        sl = slice(start, start + 50)
        col.add(ids=ids[sl], embeddings=embs[sl], documents=docs[sl], metadatas=metas[sl])
    return col.count()


def build_simple(client, rng, name, n, configuration=None, normalize=False):
    col = fresh(client, name, configuration=configuration) if configuration else fresh(client, name)
    if n:
        e = [unit(vec(rng)) if normalize else vec(rng, rng.uniform(0.2, 5.0)) for _ in range(n)]
        col.add(ids=[f"{name}-{i}" for i in range(n)], embeddings=e,
                documents=[text(rng, 10) for _ in range(n)],
                metadatas=[{"group": f"g{i % 4}", "rank": i} for i in range(n)])
    return col.count()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma-path", required=True)
    args = ap.parse_args()
    rng = random.Random(42)
    client = chromadb.PersistentClient(path=args.chroma_path)
    print("qa_main", build_main(client, rng))
    print("qa_cosine", build_simple(client, rng, "qa_cosine", 60, {"hnsw": {"space": "cosine"}}, normalize=True))
    print("qa_ip", build_simple(client, rng, "qa_ip", 60, {"hnsw": {"space": "ip"}}))
    print("qa_empty", build_simple(client, rng, "qa_empty", 0))
    blocked = fresh(client, "qa_blocked")
    blocked.add(ids=["ok-1", "zero", "huge"], embeddings=[vec(rng), [0.0] * DIM, vec(rng)],
                documents=["fine", "zero vector", "h" * 45000], metadatas=[{"a": 1}, {"a": 2}, {"a": 3}])
    print("qa_blocked", blocked.count())
    fts = fresh(client, "qa_fts")
    fts.add(ids=[f"fts-{i}" for i in range(20)], embeddings=[vec(rng) for _ in range(20)],
            documents=["pineconevectors " * 3750] + [None, ""] + [text(rng, 15) for _ in range(17)],
            metadatas=[{"n": i} for i in range(20)])
    print("qa_fts", fts.count())
    edge = fresh(client, "qa_tok_edge")
    edge.add(ids=["tok-10000", "tok-1024b", "tok-ok"], embeddings=[vec(rng) for _ in range(3)],
             documents=["a " * 10000, "c" * 1024 + " tail", "short text"])
    print("qa_tok_edge", edge.count())
    over = fresh(client, "qa_tok_over")
    over.add(ids=["tok-30000", "tok-1025b", "tok-ok"], embeddings=[vec(rng) for _ in range(3)],
             documents=["a " * 30000, "b" * 1025 + " tail", "short text"])
    print("qa_tok_over", over.count())


if __name__ == "__main__":
    main()
