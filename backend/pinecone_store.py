"""Pinecone Documents API storage (schema indexes, API version 2026-07+)."""

import os
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

DOCS_NS = "documents"
CHUNKS_NS = "chunks"
_UPSERT_BATCH = 50

_PC = None
_INDEX = None
_SCHEMA = None

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


class _Schema:
    def __init__(self):
        self.embed_field: Optional[str] = None
        self.dense_field: Optional[str] = None
        self.text_fields: list = []
        self.dense_dim: Optional[int] = None
        self.embed_model: Optional[str] = None
        self.raw_fields: dict = {}


def store_uri() -> str:
    return (os.getenv("PINECONE_INDEX") or "one-third").strip().strip("'\"")


def embed_model_name() -> str:
    schema = _schema()
    if schema.embed_model:
        return schema.embed_model
    return (os.getenv("PINECONE_EMBED_MODEL") or "llama-text-embed-v2").strip() or "llama-text-embed-v2"


def embedding_dim() -> int:
    schema = _schema()
    if schema.dense_dim:
        return int(schema.dense_dim)
    raw = (os.getenv("PINECONE_DIMENSION") or "").strip()
    return int(raw) if raw.isdigit() else 1024


def uses_integrated_embed() -> bool:
    return bool(_schema().embed_field)


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


def _field_map(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "items"):
        try:
            return dict(obj.items())
        except Exception:
            pass
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return getattr(obj, "__dict__", {}) or {}


def _parse_schema(desc) -> _Schema:
    parsed = _Schema()
    schema = getattr(desc, "schema", None)
    if schema is None and isinstance(desc, dict):
        schema = desc.get("schema")
    fields = _field_map(schema).get("fields") or _field_map(getattr(schema, "fields", None))
    if not isinstance(fields, dict):
        fields = _field_map(fields)
    parsed.raw_fields = fields

    for name, spec in fields.items():
        info = _field_map(spec)
        kind = (info.get("type") or info.get("field_type") or "").lower()
        embed = info.get("embed")
        sparse_embed = info.get("sparse_embed")
        fts = info.get("full_text_search")
        if kind == "dense_vector" or info.get("dimension"):
            parsed.dense_field = parsed.dense_field or name
            dim = info.get("dimension")
            if dim:
                parsed.dense_dim = int(dim)
        if embed:
            parsed.embed_field = parsed.embed_field or name
            embed_info = _field_map(embed)
            parsed.embed_model = embed_info.get("model") or parsed.embed_model
            parsed.text_fields.append(name)
        elif kind == "string" or fts or sparse_embed:
            parsed.text_fields.append(name)
    return parsed


def _client():
    global _PC
    if _PC is not None:
        return _PC
    api_key = _api_key()
    if not api_key:
        raise RuntimeError(
            "Pinecone is not configured. Set PINECONE_API_KEY in backend/.env "
            "(local) or the Render environment variables."
        )
    from pinecone import Pinecone

    _PC = Pinecone(api_key=api_key)
    return _PC


def _ensure_index():
    pc = _client()
    name = store_uri()
    existing = _index_names(pc)
    if name in existing:
        return
    from pinecone import SchemaBuilder

    model = (os.getenv("PINECONE_EMBED_MODEL") or "llama-text-embed-v2").strip()
    try:
        schema = (
            SchemaBuilder()
            .add_string_field("text", embed={"model": model})
            .build()
        )
        pc.indexes.create(name=name, schema=schema)
        print(f"✅ Created Pinecone document index '{name}' with embed field 'text'")
    except Exception as e:
        hint = f" Existing indexes: {existing}." if existing else ""
        raise RuntimeError(
            f"Could not create Pinecone index '{name}': {e}.{hint} "
            "Set PINECONE_INDEX to your existing document index (one-third)."
        ) from e


def _index():
    global _INDEX, _SCHEMA
    if _INDEX is not None:
        return _INDEX
    pc = _client()
    _ensure_index()
    name = store_uri()
    _INDEX = pc.index(name)
    try:
        desc = pc.indexes.describe(name)
    except Exception:
        desc = pc.describe_index(name)
    _SCHEMA = _parse_schema(desc)
    print(
        f"✅ Pinecone documents index '{name}' "
        f"embed={_SCHEMA.embed_field!r} dense={_SCHEMA.dense_field!r} "
        f"text_fields={_SCHEMA.text_fields}"
    )
    return _INDEX


def _schema() -> _Schema:
    if _SCHEMA is None:
        _index()
    return _SCHEMA or _Schema()


def _docs():
    return _index().documents


def _as_dict(doc) -> dict:
    if doc is None:
        return {}
    if isinstance(doc, dict):
        return dict(doc)
    for method in ("model_dump", "to_dict", "dict"):
        if hasattr(doc, method):
            try:
                data = getattr(doc, method)()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    data = {}
    for key in dir(doc):
        if key.startswith("_") and key not in ("_id",):
            continue
        if key in ("score",):
            data[key] = getattr(doc, key)
            continue
        try:
            value = getattr(doc, key)
        except Exception:
            continue
        if callable(value):
            continue
        data[key] = value
    if hasattr(doc, "id") and "id" not in data:
        data["id"] = doc.id
    return data


def _doc_id(doc_id, doc) -> str:
    data = _as_dict(doc)
    return str(doc_id or data.get("id") or data.get("_id") or "")


def embed_texts(texts: list, input_type: str = "passage") -> list:
    """Kept for dense-vector schemas. Integrated-embed indexes skip this."""
    if uses_integrated_embed():
        return [[] for _ in texts]
    raise RuntimeError(
        "This Pinecone index does not declare an integrated embed field, and "
        "the Documents API should be used with EmbedQuery when it does. "
        f"Schema fields: {list(_schema().raw_fields)}"
    )


def ping_store() -> dict:
    index = _index()
    schema = _schema()
    namespaces = []
    try:
        stats = index.describe_index_stats()
        raw = getattr(stats, "namespaces", None) or (stats.get("namespaces") if isinstance(stats, dict) else {}) or {}
        namespaces = list(raw.keys()) if hasattr(raw, "keys") else []
    except Exception:
        pass
    return {
        "index": store_uri(),
        "api": "documents",
        "embed_field": schema.embed_field,
        "dense_field": schema.dense_field,
        "text_fields": schema.text_fields,
        "embed_model": embed_model_name(),
        "dimension": embedding_dim(),
        "namespaces": namespaces,
    }


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


def _fetch_page(namespace: str, **kwargs):
    return _docs().fetch(namespace=namespace, **kwargs)


def _fetch_all(namespace: str, filter: Optional[dict] = None) -> list:
    rows = []
    if filter:
        token = None
        while True:
            res = _fetch_page(namespace=namespace, filter=filter, pagination_token=token)
            docs = getattr(res, "documents", None) or {}
            for doc_id, doc in docs.items():
                rows.append((_doc_id(doc_id, doc), _as_dict(doc)))
            pagination = getattr(res, "pagination", None)
            token = getattr(pagination, "next", None) if pagination else None
            if not token:
                break
        return rows

    ids = []
    for record in _docs().list(namespace=namespace):
        ids.append(getattr(record, "id", None) or record.get("id"))
        if len(ids) >= 100:
            res = _fetch_page(namespace=namespace, ids=ids)
            docs = getattr(res, "documents", None) or {}
            for doc_id, doc in docs.items():
                rows.append((_doc_id(doc_id, doc), _as_dict(doc)))
            ids = []
    if ids:
        res = _fetch_page(namespace=namespace, ids=ids)
        docs = getattr(res, "documents", None) or {}
        for doc_id, doc in docs.items():
            rows.append((_doc_id(doc_id, doc), _as_dict(doc)))
    return rows


def _count_namespace(namespace: str, ready_only: bool = False) -> Optional[int]:
    try:
        total = 0
        for _ in _docs().list(namespace=namespace):
            total += 1
        if ready_only:
            ready = 0
            for _, meta in _fetch_all(namespace):
                if (meta.get("status") or "") == "ready":
                    ready += 1
            return ready
        return total
    except Exception as e:
        print(f"⚠️ {namespace} count failed: {e}")
        return None


def count_ready_chunks() -> Optional[int]:
    return _count_namespace(CHUNKS_NS, ready_only=False)


def count_ready_documents() -> Optional[int]:
    return _count_namespace(DOCS_NS, ready_only=True)


def list_documents() -> list:
    rows = [_public_document(doc_id, meta) for doc_id, meta in _fetch_all(DOCS_NS)]
    rows.sort(key=lambda row: row.get("created_at") or "", reverse=True)
    return rows


def get_document_by_filename(filename: str) -> Optional[dict]:
    try:
        res = _fetch_page(namespace=DOCS_NS, filter={"filename": {"$eq": filename}})
        docs = getattr(res, "documents", None) or {}
        if not docs:
            return None
        doc_id, doc = next(iter(docs.items()))
        return _public_document(_doc_id(doc_id, doc), _as_dict(doc))
    except Exception as e:
        print(f"⚠️ get_document_by_filename failed: {e}")
        for doc_id, meta in _fetch_all(DOCS_NS):
            if meta.get("filename") == filename:
                return _public_document(doc_id, meta)
        return None


def _searchable_text(fields: dict) -> str:
    schema = _schema()
    title = fields.get("title") or ""
    filename = fields.get("filename") or ""
    return f"{title} {filename}".strip() or filename or "document"


def insert_document(fields: dict) -> str:
    document_id = str(uuid.uuid4())
    now = _now()
    doc = {
        "_id": document_id,
        "title": fields["title"],
        "filename": fields["filename"],
        "checksum": fields.get("checksum") or "",
        "status": fields.get("status") or "processing",
        "page_count": int(fields.get("page_count") or 0),
        "chunk_count": int(fields.get("chunk_count") or 0),
        "embedding_model": fields.get("embedding_model") or embed_model_name(),
        "error": fields.get("error") or "",
        "created_at": now,
        "updated_at": now,
    }
    schema = _schema()
    if schema.embed_field:
        doc[schema.embed_field] = _searchable_text(fields)
    _docs().upsert(namespace=DOCS_NS, documents=[doc])
    return document_id


def _get_document(document_id: str) -> Optional[dict]:
    res = _fetch_page(namespace=DOCS_NS, ids=[document_id])
    docs = getattr(res, "documents", None) or {}
    doc = docs.get(document_id)
    if doc is None and docs:
        doc_id, doc = next(iter(docs.items()))
        return _public_document(_doc_id(doc_id, doc), _as_dict(doc))
    if doc is None:
        return None
    return _public_document(document_id, _as_dict(doc))


def update_document(document_id: str, fields: dict) -> None:
    current = _get_document(document_id)
    if current is None:
        raise DocumentNotFound(document_id)
    patch = {"_id": document_id}
    allowed = {
        "title",
        "checksum",
        "status",
        "page_count",
        "chunk_count",
        "embedding_model",
        "error",
    }
    for key in allowed:
        if key in fields:
            value = fields[key]
            if key == "error" and value is None:
                value = ""
            if key in ("page_count", "chunk_count") and value is not None:
                value = int(value)
            patch[key] = value
    schema = _schema()
    if schema.embed_field and ("title" in patch or "filename" in fields):
        merged = {**current, **patch}
        patch[schema.embed_field] = _searchable_text(merged)
    _docs().update(namespace=DOCS_NS, documents=[patch])


def delete_chunks(document_id: str) -> None:
    _docs().delete(namespace=CHUNKS_NS, filter={"document_id": {"$eq": document_id}})


def delete_document(document_id: str) -> dict:
    current = _get_document(document_id)
    if current is None:
        raise DocumentNotFound(document_id)
    delete_chunks(document_id)
    _docs().delete(namespace=DOCS_NS, ids=[document_id])
    return {"deleted": True, "id": document_id, "filename": current.get("filename")}


def replace_chunks(
    document_id: str,
    filename: str,
    title: str,
    rows: list,
    batch_size: int = 50,
    replace: bool = True,
) -> None:
    schema = _schema()
    payload = []
    for row in rows:
        text = row["text"]
        doc = {
            "_id": str(uuid.uuid4()),
            "document_id": document_id,
            "chunk_index": int(row["chunk_index"]),
            "page": int(row["page"]),
            "text": text,
            "embedding_model": row.get("embedding_model") or embed_model_name(),
            "title": title or filename,
            "filename": filename,
            "status": "ready",
        }
        if schema.embed_field and schema.embed_field != "text":
            doc[schema.embed_field] = text
        if schema.dense_field:
            vector = row.get("vector") or []
            if not vector:
                raise ValueError(
                    f"Schema requires dense field '{schema.dense_field}' but no vector was provided."
                )
            doc[schema.dense_field] = [float(x) for x in vector]
        payload.append(doc)

    if replace:
        try:
            _docs().delete(namespace=CHUNKS_NS, filter={"document_id": {"$eq": document_id}})
        except Exception as e:
            print(f"⚠️ Pinecone chunk delete before replace: {e}")

    if not payload:
        return

    upsert = getattr(_docs(), "batch_upsert", None)
    if callable(upsert):
        result = upsert(
            namespace=CHUNKS_NS,
            documents=payload,
            batch_size=min(max(batch_size, 1), _UPSERT_BATCH),
            show_progress=False,
        )
        failed = getattr(result, "failed_item_count", 0) or 0
        if failed:
            errors = getattr(result, "errors", None) or []
            raise RuntimeError(f"Pinecone batch upsert failed for {failed} chunks: {errors[:2]}")
        return

    step = min(max(batch_size, 1), _UPSERT_BATCH)
    for start in range(0, len(payload), step):
        _docs().upsert(namespace=CHUNKS_NS, documents=payload[start:start + step])


def search_chunks(vector: list, top_k: int, threshold: float, query_text: str = "") -> list:
    schema = _schema()
    score_by = None
    if schema.embed_field and query_text:
        score_by = [{"type": "embed", "field": schema.embed_field, "query": query_text}]
    elif schema.dense_field and vector:
        score_by = [{"type": "dense_vector", "field": schema.dense_field, "values": [float(x) for x in vector]}]
    elif schema.text_fields and query_text:
        score_by = [{"type": "text", "query": query_text, "fields": schema.text_fields}]
    else:
        print("⚠️ Pinecone schema has no searchable field for this query")
        return []

    try:
        res = _docs().search(
            namespace=CHUNKS_NS,
            top_k=top_k,
            score_by=score_by,
            include_fields=["*"],
            filter={"status": {"$eq": "ready"}},
        )
    except Exception as e:
        print(f"⚠️ Pinecone documents search with status filter failed: {e}")
        try:
            res = _docs().search(
                namespace=CHUNKS_NS,
                top_k=top_k,
                score_by=score_by,
                include_fields=["*"],
            )
        except Exception as e2:
            print(f"⚠️ Pinecone documents search failed: {e2}")
            return []

    hits = []
    for match in getattr(res, "matches", None) or []:
        score = float(getattr(match, "score", None) or getattr(match, "_score", None) or 0.0)
        if score >= threshold:
            hits.append(_hit(_as_dict(match), score))
    return hits


def keyword_chunks(question: str, top_k: int) -> list:
    tokens = re.findall(r"[A-Za-z0-9]{3,}", (question or "").lower())[:5]
    if not tokens:
        return []
    schema = _schema()
    fields = schema.text_fields or (["text"] if schema.embed_field == "text" else [])
    query = " ".join(tokens)
    if fields:
        try:
            res = _docs().search(
                namespace=CHUNKS_NS,
                top_k=top_k,
                score_by=[{"type": "text", "query": query, "fields": fields}],
                include_fields=["*"],
            )
            return [_hit(_as_dict(match), float(getattr(match, "score", None) or 0.0))
                    for match in (getattr(res, "matches", None) or [])]
        except Exception as e:
            print(f"⚠️ Keyword documents search failed: {e}")

    hits = []
    try:
        for _, meta in _fetch_all(CHUNKS_NS):
            text = (meta.get("text") or "").lower()
            if any(token in text for token in tokens):
                hits.append(_hit(meta, 0.0))
            if len(hits) >= top_k:
                break
    except Exception as e:
        print(f"⚠️ Keyword fallback failed: {e}")
    return hits


def _hit(meta: dict, score: float) -> dict:
    title = meta.get("title") or meta.get("filename") or "Document"
    filename = meta.get("filename") or title
    return {
        "text": meta.get("text") or meta.get(_schema().embed_field or "") or "",
        "page": meta.get("page") or 0,
        "source": filename,
        "title": title,
        "document_id": meta.get("document_id"),
        "score": float(score),
    }
