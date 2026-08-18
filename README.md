# Asistente Médico RAG — MedQuAD

Sistema de preguntas y respuestas médicas sobre el corpus **MedQuAD**, con recuperación semántica (RAG), guardrails de seguridad e interfaz web.

## Stack

- **FastAPI** — API REST (`POST /ask`) y documentación automática (`/docs`)
- **Gradio** — interfaz visual, montada sobre el mismo proceso FastAPI (sirve en `/`)
- **LangChain** — orquestación del retriever y el LLM
- **OpenAI** — `text-embedding-3-small` (embeddings) y `gpt-4o-mini` (generación)
- **Supabase + pgvector** — base vectorial con 38,127 chunks (estrategia de chunking `B_512`)

## Cómo funciona

1. La pregunta pasa por los guardrails de entrada: prompt injection y toxicidad bloquean; PII se enmascara.
2. Se convierte en un vector de 1536 dimensiones y se buscan los 5 chunks más similares en Supabase (similitud coseno vía `match_documents`, con índice IVFFlat).
3. `gpt-4o-mini` responde usando **solo** ese contexto; si no está ahí, responde "I don't know".
4. La respuesta pasa por los guardrails de salida: filtro de toxicidad y disclaimer clínico obligatorio.
5. Se devuelven la respuesta y las citas de las fuentes originales.

## Correr localmente

```bash
docker build -t medquad-rag .
docker run -p 7860:7860 --env-file .env medquad-rag
```

Luego abrir http://localhost:7860 (interfaz) o http://localhost:7860/docs (API).

Variables de entorno necesarias (ver `.env.example`): `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`.

## Estructura

```
app/
  main.py         FastAPI + Gradio en un solo proceso
  rag_chain.py    retriever (Supabase) + LLM
  guardrails.py   los 4 guardrails
  schemas.py      modelos Pydantic
  config.py       variables de entorno
scripts/
  ingest_to_supabase.py   ingesta del parquet a Supabase (se corre una sola vez)
sql/
  schema.sql      tabla documents + índice ivfflat + función match_documents
```

## Aviso

Herramienta con fines educativos. No reemplaza el criterio de un profesional médico matriculado.
