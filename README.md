# Asistente Médico RAG — MedQuAD

Chat de preguntas y respuestas médicas **en español** sobre el corpus **MedQuAD** de los Institutos Nacionales de Salud de EE. UU., con recuperación semántica (RAG), memoria conversacional, guardrails de seguridad y citación de fuentes.

**Integrantes:** Castro Lozano, Ernesto Saniel · Quispe Bernardo, Andrés

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
   ├─► guardrails de entrada   inyección y toxicidad bloquean; el PII se elimina
   ├─► saludos y meta-preguntas se responden sin tocar el RAG
   │
   ├─► historial (últimas 3)   desde Supabase, por conversación
   ├─► reformulación           español + historial ──► consulta en inglés autocontenida
   │
   ├─► búsqueda vectorial      20 candidatos por similitud coseno
   ├─► corte relativo          descarta lo que cae bajo el 80% del mejor
   ├─► reranking               el LLM elige cuáles sirven de verdad
   │
   ├─► generación              contexto en inglés ──► respuesta en español
   ├─► guardrails de salida    toxicidad + disclaimer clínico
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

La consulta generada es además **impersonal**. *«¿Qué causa mis dolores de cabeza?»* se busca como `"what causes headaches?"`, y *«¿mi hijo tendrá asma?»* como `"is asthma hereditary risk factors"`. Conservar el posesivo hacía que el reranker descartara todos los pasajes — ninguno habla de esa persona en particular — y el sistema respondía "no hay información" sobre temas que sí están en el corpus.

### Memoria por usuario

El identificador es un UUID que el navegador guarda en `localStorage`, **no la IP**. Detrás de un NAT todos los usuarios de una red comparten IP pública — en una demo con varios evaluadores en el mismo wifi compartirían historial —, cambia si es dinámica, en Render llega la del proxy, y además es un dato personal: guardarla sería incoherente con tener un guardrail que elimina PII.

Cada usuario puede tener varias conversaciones, listadas en la barra lateral. Lo que se guarda pasa antes por `strip_pii`.

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

134 pruebas. Ver [tests/README.md](tests/README.md) para el detalle de qué cubre cada una y qué regresión previene.

## Notebooks — la fase de investigación

En [`notebooks/`](notebooks/) está el trabajo previo que llevó a esta arquitectura. Se conservan tal como se ejecutaron: son la evidencia de que la configuración desplegada se eligió comparando alternativas, no por intuición.

| Notebook | Contenido |
|---|---|
| [01 — Data preparation and cleaning](notebooks/01_data_preparation_and_cleaning.ipynb) | Carga y exploración de MedQuAD, diagnóstico y limpieza del dataset |
| [02 — Chunking, embeddings and indexing](notebooks/02_chunking_embeddings_and_indexing.ipynb) | Comparación de tamaños de chunk, generación de embeddings, indexación con FAISS y BM25 |
| [03 — RAG inference pipeline](notebooks/03_rag_inference_pipeline.ipynb) | Pipeline completo, citación obligatoria, grounding y los cuatro guardrails |
| [04 — Evaluation and benchmarks](notebooks/04_evaluation_and_benchmarks.ipynb) | Recall@k y Precision@k del retrieval, y evaluación de la generación |
| [05 — User interface with Gradio](notebooks/05_user_interface_gradio.ipynb) | Primera interfaz web sobre el pipeline |

### Qué cambió del notebook a producción

| | Notebook (investigación) | Producción |
|---|---|---|
| Embeddings | `all-mpnet-base-v2` local | OpenAI `text-embedding-3-small` a 512 dims |
| Índice | FAISS en memoria | Postgres + pgvector (IVFFlat) en Supabase |
| Chunking | 3 estrategias comparadas | solo `B_512`, la ganadora |
| LLM | Qwen2.5-7B local | `gpt-4o-mini` |
| Idioma | preguntas en inglés | preguntas en español, corpus en inglés |
| Conversación | preguntas sueltas | memoria de las últimas 3 interacciones |

## Evaluación del retrieval

Las decisiones sobre el pipeline —descartar búsqueda híbrida, calibrar el corte relativo, activar el reranking— se tomaron midiendo. Los scripts y resultados están en [eval/README.md](eval/README.md), junto con la advertencia de qué métrica resultó engañosa y por qué.

## Despliegue

El `Dockerfile` escucha en `${PORT}` si la plataforma lo define, así que funciona tal cual en Render, Fly.io o cualquier servicio que acepte contenedores.

En Render: **New → Web Service**, conectar el repositorio (detecta el `Dockerfile` solo), cargar las tres variables de entorno y desplegar. En el plan gratuito el servicio se duerme tras 15 minutos sin uso y el primer request después tarda cerca de un minuto.

## Estructura

```
app/
  main.py         FastAPI + orquestación del turno de chat
  ui.py           interfaz Gradio: barra lateral y chat
  rag_chain.py    reformulación, retriever, reranking y generación
  memory.py       usuarios, conversaciones e historial en Supabase
  guardrails.py   los 4 guardrails, en español e inglés
  citations.py    formato de las fuentes y rescate de enlaces caídos
  conversacion.py saludos y preguntas sobre el asistente
  schemas.py      modelos Pydantic
  config.py       variables de entorno
notebooks/        fase de investigación (ver arriba)
eval/             evaluación del retrieval (LLM-as-judge)
tests/            unitarios e integración
scripts/          ingesta del parquet a Supabase (se corre una sola vez)
sql/schema.sql    documents + ivfflat + match_documents + chat_history + conversations
```

## Aviso

Herramienta con fines educativos. No reemplaza el criterio de un profesional médico colegiado.
