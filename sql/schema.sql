-- Esquema de la base vectorial en Supabase (pgvector).

create extension if not exists vector;

-- La dimensión debe coincidir con EMBEDDING_DIMENSIONS en
-- scripts/ingest_to_supabase.py y con OpenAIEmbeddings(dimensions=...) en
-- app/rag_chain.py. Si las tres no están alineadas, pgvector rechaza la
-- consulta por dimensión incompatible.
--
-- Por qué 512 y no las 1536 por defecto: con 1536 la base llegaba a 643 MB
-- contra el límite de 500 MB del free tier (93% del espacio eran vectores,
-- no datos). text-embedding-3-small admite truncar el vector con el parámetro
-- `dimensions` conservando la mayor parte de la información semántica.
create table documents (
  id bigserial primary key,
  content text,             -- el chunk_text que se le pasa al LLM como contexto
                            -- (el embedding se calcula sobre question + chunk_text,
                            -- ver scripts/ingest_to_supabase.py — content no cambia)
  metadata jsonb,           -- question, document_source, document_url, question_focus,
                            -- document_id, chunk_id, n_chunks
  embedding vector(512)     -- text-embedding-3-small truncado a 512
);

-- ------------------------------------------------------------
-- Índice IVFFlat — correr DESPUÉS de la ingesta (necesita los datos presentes
-- para calcular sus centroides).
--
-- Sin él, pgvector hace búsqueda exhaustiva exacta sobre los 38,127 vectores,
-- lo que excede el statement_timeout del free tier y aborta con
-- `57014 canceling statement due to statement timeout`. Medido: sin índice la
-- búsqueda tardaba entre 0.5 s y 5.6 s con timeouts intermitentes; con índice,
-- 0.34 s de media.
--
-- Los dos `set` deben correrse en la MISMA sesión que el create index:
--   · maintenance_work_mem: la construcción toma una muestra de 50 × lists
--     vectores para el k-means. Con 1536 dims eran ~70 MB y fallaba con
--     `54000 memory required is 70 MB` contra el default de 32 MB; con 512
--     baja a ~24 MB y ya entraría, pero se deja explícito.
--   · statement_timeout: la construcción también está sujeta al límite de 8 s.
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
  query_embedding vector(512),
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
