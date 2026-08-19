-- Esquema de la base vectorial en Supabase (pgvector).
-- Ya aplicado en el proyecto de Supabase; se versiona como referencia.

create extension if not exists vector;

-- La dimensión debe coincidir con EMBEDDING_DIMENSIONS en
-- scripts/ingest_to_supabase.py y con OpenAIEmbeddings(dimensions=...) en
-- app/rag_chain.py. Si las tres no están alineadas, pgvector rechaza la
-- consulta por dimensión incompatible.
create table documents (
  id bigserial primary key,
  content text,             -- el chunk_text que se le pasa al LLM como contexto
                            -- (el embedding se calcula sobre question + chunk_text,
                            -- ver scripts/ingest_to_supabase.py — content no cambia)
  metadata jsonb,           -- question, document_source, document_url, question_focus,
                            -- document_id, chunk_id, n_chunks
  embedding vector(512)     -- text-embedding-3-small truncado a 512 (ver abajo)
);

-- ------------------------------------------------------------
-- Índice IVFFlat — correr DESPUÉS de la ingesta.
--
-- Sin él, pgvector hace búsqueda exhaustiva exacta sobre los 38,127 vectores,
-- lo que excede el statement_timeout del free tier de Supabase y aborta con
-- `57014 canceling statement due to statement timeout`. (Con los 1536 dims
-- originales eran ~230 MB a escanear por consulta; con 512 son ~78 MB, que
-- podría llegar a entrar en el límite, pero el índice lo baja a ~50-200 ms.)
--
-- Los dos `set` son obligatorios y deben correrse en la MISMA sesión que el
-- create index:
--   · maintenance_work_mem: la construcción toma una muestra de 50 × lists
--     vectores para el k-means. Con los 1536 dims originales eran ~70 MB y
--     fallaba con `54000 memory required is 70 MB` contra el default de 32 MB;
--     con 512 dims baja a ~24 MB y ya entraría, pero se deja explícito.
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


-- ------------------------------------------------------------
-- Historial de conversación (memoria por sesión)
--
-- `session_id` es un UUID que el navegador guarda en localStorage
-- (gr.BrowserState), NO la IP. Detrás de un NAT todos los usuarios de una red
-- comparten IP pública —en una demo con varios evaluadores en el mismo wifi
-- compartirían historial—, cambia si es dinámica, en Render llega la del proxy
-- y no la del usuario, y además es un dato personal: guardarla sería
-- incoherente con tener un guardrail que enmascara PII.
--
-- El `content` del rol 'user' se guarda ya sin PII (guardrails.strip_pii).
-- ------------------------------------------------------------
create table chat_history (
  id bigserial primary key,
  session_id text not null,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  created_at timestamptz not null default now()
);

-- La consulta de cada turno es "últimos N mensajes de esta sesión".
create index chat_history_session_idx on chat_history (session_id, created_at desc);

-- Retención: el historial crece sin límite y el free tier son 0.5 GB. Con la
-- base al 49% no es urgente, pero conviene purgar cada tanto:
--   delete from chat_history where created_at < now() - interval '30 days';
