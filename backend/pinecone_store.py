"""Pinecone storage for documents and embedded chunks."""

import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

EMBEDDING_DIM = 384
DOCS_NS = "documents"
CHUNKS_NS = "chunks"
_DOC_VECTOR = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
_UPSERT_BATCH = 100

_INDEX = None

_DOC_FIELDS = (
    "id",
    "title",
    "filename",
    "checksum",
    "status",
    "page_count",
    "chunk_count",
    "embedding_model",
    "error",
    "created_at",
    "updated_at",
)


class DocumentNotFound(Exception):
    pass


def store_uri() -> str:
    return (os.getenv("PINECONE_INDEX") or "document-qa").strip().strip("'\"")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _api_key() -> str:
    return (os.getenv("PINECONE_API_KEY") or "").strip().strip("'\"")


def _index_names(pc) -> list:
    listed = pc.list_indexes()
    if hasattr(listed, "names"):
        try:
            names = list(listed.names())
            if names:
                return names
        except TypeError:
            pass
    items = listed if isinstance(listed, list) else getattr(listed, "indexes", None) or []
    names = []
    for item in items:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and item.get("name"):
            names.append(item["name"])
        elif getattr(item, "name", None):
            names.append(item.name)
    return names


def _index_field(desc, *keys, default=None):
    current = desc
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return default if current is None else current


def _connect_index(pc, name: str):
    desc = pc.describe_index(name)
    host = _index_field(desc, "host")
    dimension = int(_index_field(desc, "dimension", default=0) or 0)
    if dimension and dimension != EMBEDDING_DIM:
        raise RuntimeError(
            f"Pinecone index '{name}' is {dimension}-dimensional, but embeddings "
            f"are {EMBEDDING_DIM}-d. Create a cosine index with dimension {EMBEDDING_DIM} "
            f"or set PINECONE_INDEX to that index."
        )
    return pc.Index(host=host) if host else pc.Index(name)


def _index():
    global _INDEX
    if _INDEX is not None:
        return _INDEX

    api_key = _api_key()
    if not api_key:
        raise RuntimeError(
            "Pinecone is not configured. Set PINECONE_API_KEY in backend/.env "
            "(local) or the Render environment variables."
        )

    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=api_key)
    host = (os.getenv("PINECONE_HOST") or "").strip().strip("'\"")
    name = store_uri()

    if host:
        _INDEX = pc.Index(host=host)
        return _INDEX

    existing = _index_names(pc)
    if name not in existing:
        cloud = (os.getenv("PINECONE_CLOUD") or "aws").strip()
        region = (os.getenv("PINECONE_REGION") or "us-east-1").strip()
        try:
            pc.create_index(
                name=name,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        except Exception as e:
            hint = f" Existing indexes: {existing}." if existing else ""
            raise RuntimeError(
                f"Could not create Pinecone index '{name}': {e}.{hint} "
                "On the free plan you usually get one index. Set PINECONE_INDEX "
                f"to an existing cosine {EMBEDDING_DIM}-d index."
            ) from e
        for _ in range(60):
            desc = pc.describe_index(name)
            if _index_field(desc, "status", "ready"):
                break
            time.sleep(1)

    _INDEX = _connect_index(pc, name)
    return _INDEX


def ping_store() -> dict:
    """Open the index once so ingest can fail with a single clear error."""
    index = _index()
    stats = index.describe_index_stats()
    namespaces = {}
    if isinstance(stats, dict):
        namespaces = stats.get("namespaces") or {}
    else:
        namespaces = getattr(stats, "namespaces", None) or {}
    return {
        "index": store_uri(),
        "namespaces": list(namespaces.keys()) if hasattr(namespaces, "keys") else [],
    }


def _vectors_from_fetch(result) -> dict:
    if hasattr(result, "vectors"):
        return result.vectors or {}
    if isinstance(result, dict):
        return result.get("vectors") or {}
    return {}


def _meta(record) -> dict:
    if record is None:
        return {}
    if hasattr(record, "metadata"):
        return record.metadata or {}
    if isinstance(record, dict):
        return record.get("metadata") or {}
    return {}


def _public_document(document_id: str, meta: dict) -> dict:
    item = {key: meta.get(key) for key in _DOC_FIELDS}
    item["id"] = document_id
    if item.get("error") == "":
        item["error"] = None
    for key in ("page_count", "chunk_count"):
        if item.get(key) is not None:
            try:
                item[key] = int(item[key])
            except (TypeError, ValueError):
                item[key] = 0
    return item


def _iter_ids(namespace: str):
    index = _index()
    for page in index.list(namespace=namespace):
        ids = list(page or [])
        if ids:
            yield ids


def _fetch_all(namespace: str) -> list:
    records = []
    index = _index()
    for ids in _iter_ids(namespace):
        fetched = _vectors_from_fetch(index.fetch(ids=ids, namespace=namespace))
        for vector_id, record in fetched.items():
            records.append((vector_id, _meta(record)))
    return records


def count_ready_chunks() -> Optional[int]:
    try:
        stats = _index().describe_index_stats()
        namespaces = getattr(stats, "namespaces", None) or stats.get("namespaces") or {}
        info = namespaces.get(CHUNKS_NS) or {}
        if hasattr(info, "vector_count"):
            return int(info.vector_count or 0)
        return int(info.get("vector_count") or 0)
    except Exception as e:
        print(f"⚠️ chunks count failed: {e}")
        return None


def count_ready_documents() -> Optional[int]:
    try:
        ready = 0
        for _, meta in _fetch_all(DOCS_NS):
            if (meta.get("status") or "") == "ready":
                ready += 1
        return ready
    except Exception as e:
        print(f"⚠️ documents count failed: {e}")
        return None


def list_documents() -> list:
    rows = [_public_document(vector_id, meta) for vector_id, meta in _fetch_all(DOCS_NS)]
    rows.sort(key=lambda row: row.get("created_at") or "", reverse=True)
    return rows


def get_document_by_filename(filename: str) -> Optional[dict]:
    try:
        res = _index().query(
            namespace=DOCS_NS,
            vector=_DOC_VECTOR,
            top_k=20,
            filter={"filename": {"$eq": filename}},
            include_metadata=True,
        )
        matches = getattr(res, "matches", None) or res.get("matches") or []
    except Exception:
        matches = []
        for vector_id, meta in _fetch_all(DOCS_NS):
            if meta.get("filename") == filename:
                return _public_document(vector_id, meta)
    if not matches:
        return None
    match = matches[0]
    vector_id = getattr(match, "id", None) or match.get("id")
    return _public_document(vector_id, _meta(match))


def insert_document(fields: dict) -> str:
    document_id = str(uuid.uuid4())
    now = _now()
    meta = {
        "title": fields["title"],
        "filename": fields["filename"],
        "checksum": fields.get("checksum") or "",
        "status": fields.get("status") or "processing",
        "page_count": int(fields.get("page_count") or 0),
        "chunk_count": int(fields.get("chunk_count") or 0),
        "embedding_model": fields.get("embedding_model") or "",
        "error": fields.get("error") or "",
        "created_at": now,
        "updated_at": now,
    }
    _index().upsert(
        vectors=[{"id": document_id, "values": _DOC_VECTOR, "metadata": meta}],
        namespace=DOCS_NS,
    )
    return document_id


def _get_document(document_id: str) -> Optional[dict]:
    fetched = _vectors_from_fetch(
        _index().fetch(ids=[document_id], namespace=DOCS_NS)
    )
    record = fetched.get(document_id)
    if record is None:
        return None
    return _public_document(document_id, _meta(record))


def update_document(document_id: str, fields: dict) -> None:
    current = _get_document(document_id)
    if current is None:
        raise DocumentNotFound(document_id)

    allowed = {
        "title",
        "checksum",
        "status",
        "page_count",
        "chunk_count",
        "embedding_model",
        "error",
    }
    meta = {key: current.get(key) for key in _DOC_FIELDS if key != "id"}
    for key in allowed:
        if key in fields:
            meta[key] = fields[key]
    if meta.get("error") is None:
        meta["error"] = ""
    for key in ("page_count", "chunk_count"):
        meta[key] = int(meta.get(key) or 0)
    meta["updated_at"] = _now()
    meta["created_at"] = current.get("created_at") or _now()

    _index().upsert(
        vectors=[{"id": document_id, "values": _DOC_VECTOR, "metadata": meta}],
        namespace=DOCS_NS,
    )


def delete_chunks(document_id: str) -> None:
    _index().delete(
        filter={"document_id": {"$eq": document_id}},
        namespace=CHUNKS_NS,
    )


def delete_document(document_id: str) -> dict:
    current = _get_document(document_id)
    if current is None:
        raise DocumentNotFound(document_id)
    delete_chunks(document_id)
    _index().delete(ids=[document_id], namespace=DOCS_NS)
    return {"deleted": True, "id": document_id, "filename": current.get("filename")}


def replace_chunks(
    document_id: str,
    filename: str,
    title: str,
    rows: list,
    batch_size: int = 50,
) -> None:
    payload = []
    for row in rows:
        vector = [float(x) for x in row["vector"]]
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(f"Expected embedding dim {EMBEDDING_DIM}, got {len(vector)}")
        payload.append({
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": {
                "document_id": document_id,
                "chunk_index": int(row["chunk_index"]),
                "page": int(row["page"]),
                "text": row["text"],
                "embedding_model": row.get("embedding_model") or "",
                "title": title or filename,
                "filename": filename,
                "status": "ready",
            },
        })

    index = _index()
    try:
        index.delete(filter={"document_id": {"$eq": document_id}}, namespace=CHUNKS_NS)
    except Exception as e:
        print(f"⚠️ Pinecone chunk delete before replace: {e}")

    step = min(max(batch_size, 1), _UPSERT_BATCH)
    for start in range(0, len(payload), step):
        index.upsert(vectors=payload[start:start + step], namespace=CHUNKS_NS)


def search_chunks(vector: list, top_k: int, threshold: float) -> list:
    if len(vector) != EMBEDDING_DIM:
        print(f"⚠️ Unexpected embedding dim {len(vector)} (expected {EMBEDDING_DIM})")
        return []

    try:
        res = _index().query(
            namespace=CHUNKS_NS,
            vector=[float(x) for x in vector],
            top_k=top_k,
            filter={"status": {"$eq": "ready"}},
            include_metadata=True,
        )
    except Exception as e:
        print(f"⚠️ Pinecone vector search failed: {e}")
        return []

    hits = []
    matches = getattr(res, "matches", None) or res.get("matches") or []
    for match in matches:
        score = float(getattr(match, "score", None) if hasattr(match, "score") else match.get("score") or 0.0)
        if score >= threshold:
            hits.append(_hit(_meta(match), score))
    return hits


def keyword_chunks(question: str, top_k: int) -> list:
    tokens = re.findall(r"[A-Za-z0-9]{3,}", (question or "").lower())[:5]
    if not tokens:
        return []

    try:
        hits = []
        for _, meta in _fetch_all(CHUNKS_NS):
            if (meta.get("status") or "") != "ready":
                continue
            text = (meta.get("text") or "").lower()
            if any(token in text for token in tokens):
                hits.append(_hit(meta, 0.0))
            if len(hits) >= top_k:
                break
        return hits
    except Exception as e:
        print(f"⚠️ Keyword fallback failed: {e}")
        return []


def _hit(meta: dict, score: float) -> dict:
    title = meta.get("title") or meta.get("filename") or "Document"
    filename = meta.get("filename") or title
    return {
        "text": meta.get("text") or "",
        "page": meta.get("page") or 0,
        "source": filename,
        "title": title,
        "document_id": meta.get("document_id"),
        "score": float(score),
    }
