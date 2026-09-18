-- Run this once in the Supabase SQL editor before ingesting PDFs.
create extension if not exists vector;

create table if not exists document_chunks (
  id bigserial primary key,
  source text not null,
  page integer not null default 0,
  chunk_index integer not null default 0,
  text text not null,
  embedding vector(384),
  created_at timestamptz not null default now(),
  unique (source, page, chunk_index)
);

create index if not exists document_chunks_embedding_idx
  on document_chunks
  using hnsw (embedding vector_cosine_ops);

create or replace function match_document_chunks(
  query_embedding vector(384),
  match_threshold float default 0.3,
  match_count int default 5
)
returns table (
  id bigint,
  source text,
  page integer,
  text text,
  score float
)
language sql
stable
as $$
  select
    document_chunks.id,
    document_chunks.source,
    document_chunks.page,
    document_chunks.text,
    (1 - (document_chunks.embedding <=> query_embedding))::float as score
  from document_chunks
  where 1 - (document_chunks.embedding <=> query_embedding) > match_threshold
  order by document_chunks.embedding <=> query_embedding
  limit match_count;
$$;
