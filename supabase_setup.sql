create extension if not exists vector;

create table if not exists gog_documents (
  id text primary key,
  source text not null,
  type text not null,
  title text not null,
  url text,
  content text not null,
  chunk_index integer default 0,
  total_chunks integer default 1,
  embedding vector(1536),
  created_at timestamp with time zone default now()
);

create index if not exists gog_documents_embedding_idx
  on gog_documents
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 20);

create or replace function match_gog_documents(
  query_embedding vector(1536),
  match_threshold float default 0.5,
  match_count int default 5
)
returns table (
  id text,
  source text,
  type text,
  title text,
  url text,
  content text,
  similarity float
)
language sql stable
as $$
  select
    id,
    source,
    type,
    title,
    url,
    content,
    1 - (embedding <=> query_embedding) as similarity
  from gog_documents
  where 1 - (embedding <=> query_embedding) > match_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;
