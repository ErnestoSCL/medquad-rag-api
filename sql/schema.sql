-- Esquema de la base vectorial en Supabase (pgvector).

create extension if not exists vector;

create table documents (
  id bigserial primary key,
  content text,             -- el chunk_text que se le pasa al LLM como contexto
                            -- (el embedding se calcula sobre question + chunk_text,
                            -- ver scripts/ingest_to_supabase.py — content no cambia)
  metadata jsonb,           -- question, document_source, document_url, question_focus,
                            -- document_id, chunk_id, n_chunks
  embedding vector(1536)    -- dimensión de text-embedding-3-small
);

-- Índice IVFFlat. Sin él, pgvector hace búsqueda exhaustiva exacta sobre los
-- 38,127 vectores (~230 MB por consulta), lo que excede el statement_timeout
-- del free tier de Supabase. lists ≈ sqrt(n_filas).
-- Crear DESPUÉS de la ingesta: IVFFlat necesita los datos presentes para
-- calcular sus centroides.
set statement_timeout = '10min';

create index on documents using ivfflat (embedding vector_cosine_ops) with (lists = 195);

-- `set ivfflat.probes = 10` fija cuántas de las 195 listas se revisan por
-- consulta: más probes = más recall y más latencia. Va en la definición de la
-- función para que aplique a toda llamada desde el backend.
create function match_documents (
  query_embedding vector(1536),
  match_count int default 5
) returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language sql stable
set ivfflat.probes = 10
as $$
  select id, content, metadata,
         1 - (embedding <=> query_embedding) as similarity
  from documents
  order by embedding <=> query_embedding
  limit match_count;
$$;
