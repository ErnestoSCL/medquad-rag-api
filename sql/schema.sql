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

-- ------------------------------------------------------------
-- Índice IVFFlat — correr DESPUÉS de la ingesta.
--
-- Sin él, pgvector hace búsqueda exhaustiva exacta sobre los 38,127 vectores
-- (~230 MB por consulta), lo que excede el statement_timeout del free tier de
-- Supabase y aborta con `57014 canceling statement due to statement timeout`.
--
-- Los dos `set` son obligatorios y deben correrse en la MISMA sesión que el
-- create index:
--   · maintenance_work_mem: la construcción toma una muestra de 50 × lists
--     vectores para el k-means (195 × 50 × 1536 × 4 B ≈ 70 MB). El default del
--     free tier es 32 MB, con el que falla con `54000 memory required is 70 MB`.
--   · statement_timeout: la construcción también está sujeta al límite de 8 s.
--
-- Si 128MB tampoco alcanza, bajar `lists` (la memoria escala linealmente con
-- él): lists = 80 entra en 32 MB, a cambio de escanear ~12% del corpus en vez
-- de ~5% por consulta.
-- ------------------------------------------------------------
set maintenance_work_mem = '128MB';
set statement_timeout = '10min';

drop index if exists documents_embedding_ivfflat_idx;

create index documents_embedding_ivfflat_idx
  on documents using ivfflat (embedding vector_cosine_ops)
  with (lists = 195);

-- `set ivfflat.probes = 10` fija cuántas de las 195 listas se revisan por
-- consulta: más probes = más recall y más latencia. Va en la definición de la
-- función para que aplique a toda llamada desde el backend.
-- Sin esto, pgvector usa probes = 1 (una sola lista) y el recall cae a ~70%.
drop function if exists match_documents(vector, int);

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
