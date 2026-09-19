-- ONE-THIRD RAG schema (documents + chunks)
-- Run this once in the Supabase SQL editor.
-- Safe to re-run: uses IF NOT EXISTS / CREATE OR REPLACE.

create extension if not exists vector;
create extension if not exists pgcrypto;

-- One row per uploaded/ingested file
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  filename text not null unique,
  checksum text,
  status text not null default 'pending'
    check (status in ('pending', 'processing', 'ready', 'failed')),
  page_count integer not null default 0,
  chunk_count integer not null default 0,
  embedding_model text,
  metadata jsonb not null default '{}'::jsonb,
  error text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists documents_status_idx on documents (status);
create index if not exists documents_checksum_idx on documents (checksum);

-- Many retrieval units per document
create table if not exists chunks (
  id bigserial primary key,
  document_id uuid not null references documents(id) on delete cascade,
  chunk_index integer not null,
  page integer not null default 0,
  text text not null,
  embedding vector(384),
  embedding_model text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (document_id, chunk_index)
);

create index if not exists chunks_document_id_idx on chunks (document_id);

create index if not exists chunks_embedding_hnsw_idx
  on chunks
  using hnsw (embedding vector_cosine_ops)
  with (m = 16, ef_construction = 64);

-- Optional full-text for hybrid / keyword fallback
alter table chunks
  add column if not exists fts tsvector
  generated always as (to_tsvector('english', coalesce(text, ''))) stored;

create index if not exists chunks_fts_idx on chunks using gin (fts);

create or replace function match_chunks(
  query_embedding vector(384),
  match_threshold float default 0.05,
  match_count int default 5,
  filter_document_id uuid default null
)
returns table (
  id bigint,
  document_id uuid,
  title text,
  source text,
  page integer,
  text text,
  score float
)
language sql
stable
as $$
  select
    c.id,
    c.document_id,
    d.title,
    d.filename as source,
    c.page,
    c.text,
    (1 - (c.embedding <=> query_embedding))::float as score
  from chunks c
  join documents d on d.id = c.document_id
  where d.status = 'ready'
    and c.embedding is not null
    and (filter_document_id is null or c.document_id = filter_document_id)
    and 1 - (c.embedding <=> query_embedding) >= match_threshold
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

-- Keep updated_at fresh on document changes
create or replace function set_documents_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists documents_set_updated_at on documents;
create trigger documents_set_updated_at
  before update on documents
  for each row execute function set_documents_updated_at();

-- Optional: keep legacy document_chunks around until you re-ingest.
-- After verifying /documents and /chat work, you can drop it:
--   drop function if exists match_document_chunks(vector, float, int);
--   drop table if exists document_chunks;
