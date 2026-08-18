# Asistente Médico RAG — MedQuAD

Chat de preguntas y respuestas médicas **en español** sobre el corpus **MedQuAD**, con recuperación semántica (RAG), memoria conversacional, guardrails de seguridad y citación de fuentes.

## Stack

- **FastAPI** — API REST (`POST /ask`) y documentación automática (`/docs`)
- **Gradio** — interfaz de chat, montada sobre el mismo proceso FastAPI (sirve en `/`)
- **LangChain** — orquestación del retriever y el LLM
- **OpenAI** — `text-embedding-3-small` a 512 dimensiones y `gpt-4o-mini`
- **Supabase + pgvector** — 38,127 chunks (chunking `B_512`) con índice IVFFlat

## Cómo funciona

```
pregunta (español)
   │
   ├─► guardrails de entrada        inyección y toxicidad bloquean; el PII se elimina
   │
   ├─► historial (últimas 3)        desde Supabase, por session_id
   │
   ├─► reformulación                español + historial ──► consulta en inglés autocontenida
   │
   ├─► búsqueda vectorial           20 candidatos por similitud coseno
   ├─► corte relativo               descarta lo que cae bajo el 80% del mejor
   ├─► reranking                    el LLM elige cuáles sirven de verdad
   │
   ├─► generación                   contexto en inglés ──► respuesta en español
   │
   ├─► guardrails de salida         toxicidad + disclaimer clínico
   │
   └─► se persiste la interacción y se devuelven respuesta + fuentes
```

### Por qué se traduce la pregunta

MedQuAD está íntegramente en inglés. Buscar directo con la pregunta en español degrada la recuperación — medido sobre 8 pares equivalentes:

| | Chunks correctos | Similitud media |
|---|---|---|
| Pregunta en inglés (referencia) | 5,0 / 5 | 0,706 |
| Pregunta en español, directo | 3,0 / 5 | 0,568 |
| **Pregunta traducida** | **4,9 / 5** | **0,702** |

La misma llamada que traduce resuelve las repreguntas: con *«¿Qué es la parálisis de Bell?»* en el historial, *«¿y cómo se trata?»* se convierte en `"How is Bell's palsy treated?"`.

La consulta generada es además **impersonal**: *«¿qué causa mis dolores de cabeza?»* se busca como `"what causes headaches?"`. Conservar el posesivo hacía que el reranker descartara todos los pasajes — ninguno habla de los dolores de cabeza de esa persona en particular.

### Memoria por usuario

El `session_id` es un UUID que el navegador guarda en localStorage (`gr.BrowserState`), **no la IP**. Detrás de un NAT todos los usuarios de una red comparten IP pública — en una demo con varios evaluadores en el mismo wifi compartirían historial —, cambia si es dinámica, en Render llega la del proxy, y además es un dato personal: guardarla sería incoherente con tener un guardrail que elimina PII.

Lo que se guarda en el historial pasa antes por `strip_pii`.

## Correr localmente

```bash
docker build -t medquad-rag .
docker run -p 7860:7860 --env-file .env medquad-rag
```

Luego abrir http://localhost:7860 (chat) o http://localhost:7860/docs (API).

Variables de entorno (ver `.env.example`): `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`.

## Pruebas

```bash
pytest tests/unit -q          # sin red ni servidor
pytest tests/integration -q   # requiere el contenedor levantado
```

Ver [tests/README.md](tests/README.md) para el detalle de qué cubre cada uno.

## Evaluación del retrieval

Las decisiones sobre el pipeline (descartar búsqueda híbrida, calibrar el corte relativo, activar el reranking) se tomaron midiendo, no por intuición. Los scripts y resultados están en [eval/README.md](eval/README.md).

## Estructura

```
app/
  main.py         FastAPI + interfaz de chat Gradio
  rag_chain.py    reformulación, retriever, reranking y generación
  memory.py       historial por sesión en Supabase
  guardrails.py   los 4 guardrails, en español e inglés
  schemas.py      modelos Pydantic
  config.py       variables de entorno
eval/             evaluación del retrieval (LLM-as-judge)
tests/            unitarios e integración
scripts/
  ingest_to_supabase.py   ingesta del parquet (se corre una sola vez)
sql/
  schema.sql      documents + índice ivfflat + match_documents + chat_history
```

## Aviso

Herramienta con fines educativos. No reemplaza el criterio de un profesional médico colegiado.
