# Asistente Médico RAG — MedQuAD

Chat de preguntas y respuestas médicas **en español** sobre el corpus **MedQuAD** de los Institutos Nacionales de Salud de EE. UU., con recuperación semántica (RAG), memoria conversacional, guardrails de seguridad y citación de fuentes.

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

## Instalación y ejecución local

### Requisitos previos

- Python 3.11 o superior.
- Una clave de OpenAI.
- Un proyecto de Supabase con las tablas, funciones e índice definidos en
  [`sql/schema.sql`](sql/schema.sql).
- Git, si se va a clonar el repositorio.

### 1. Clonar el repositorio

Sustituir la URL por la URL real del repositorio en GitHub:

```powershell
git clone <URL_DEL_REPOSITORIO>
cd medquad-rag-api
```

Si el proyecto ya está descargado, basta con abrir PowerShell en la carpeta
raíz, es decir, la carpeta que contiene `README.md` y `requirements.txt`.

### 2. Crear y activar un entorno virtual

El entorno virtual mantiene las dependencias del proyecto separadas de la
instalación global de Python:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activación de scripts, permitirla solo para la
terminal actual y repetir el comando:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar las dependencias

Actualizar `pip` e instalar las dependencias de ejecución desde
[`requirements.txt`](requirements.txt):

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para ejecutar las pruebas también se necesitan las dependencias de desarrollo:

```powershell
python -m pip install -r requirements-dev.txt
```

### 4. Configurar las variables de entorno

Crear un archivo `.env` en la raíz del proyecto a partir de
[`.env.example`](.env.example) y completar:

```dotenv
OPENAI_API_KEY=tu_clave_de_openai
SUPABASE_URL=https://tu-proyecto.supabase.co
SUPABASE_KEY=tu_clave_de_supabase
```

El archivo `.env` no debe subirse a GitHub porque contiene credenciales. El
archivo `.env.example` documenta los nombres de las variables necesarias sin
exponer sus valores.

### 5. Ejecutar con Uvicorn

Desde la raíz del proyecto y con el entorno virtual activo:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Abrir en el navegador:

- Interfaz de chat: http://localhost:7860
- Documentación interactiva de la API: http://localhost:7860/docs
- Comprobación de salud: http://localhost:7860/healthz

Para detener el servidor, pulsar `Ctrl+C`. Durante el desarrollo se puede usar
`--reload` para que Uvicorn reinicie el proceso cuando cambien los archivos:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

### Ejecución con Docker

Como alternativa al entorno virtual, Docker instala las dependencias dentro de
la imagen y ejecuta la misma aplicación:

```powershell
docker build -t medquad-rag .
docker run -p 7860:7860 --env-file .env medquad-rag
```

Luego abrir http://localhost:7860. El contenedor usa el puerto `7860` por
defecto y respeta la variable `PORT` cuando una plataforma de despliegue la
define.

## Pruebas y verificación

```bash
python -m pytest tests/unit -q          # sin red ni servidor
python -m pytest tests/integration -q   # requiere la aplicación levantada
```

Las pruebas unitarias validan guardrails, conversaciones, citas, memoria y
parseo del reranker sin depender de servicios externos. Las pruebas de
integración verifican los endpoints y la persistencia contra la aplicación
ejecutándose. Ver [tests/README.md](tests/README.md) para el detalle de cada
grupo.

## Criterios del repositorio

El proyecto está organizado para que otra persona pueda clonarlo, instalarlo
y ejecutarlo sin asistencia adicional:

- **Bien estructurado:** el código Python está separado en módulos con
  responsabilidades claras: configuración, API, interfaz, lógica RAG,
  memoria, modelos y pruebas. No depende de un único archivo monolítico.
- **Documentado:** este README explica el objetivo, la arquitectura, la
  instalación, la configuración, la ejecución local, Docker, las pruebas y el
  funcionamiento del pipeline. También incluye notebooks de investigación y
  documentación específica en [`tests/README.md`](tests/README.md) y
  [`eval/README.md`](eval/README.md).
- **Variables de entorno:** las credenciales se leen desde `.env`, que debe
  mantenerse fuera del control de versiones. [`.env.example`](.env.example)
  sirve como plantilla y documenta las tres claves requeridas.
- **Listo para ejecutar:** el flujo completo es `clonar` -> `crear entorno` ->
  `instalar requirements.txt` -> `configurar .env` -> `ejecutar Uvicorn`.
  También existe un `Dockerfile` para ejecutar el proyecto sin instalar
  Python ni sus dependencias directamente en el sistema.

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
## Evidencias Funcionales

### Inicio de Chatbot
<img width="1860" height="938" alt="image" src="https://github.com/user-attachments/assets/0a2fb1d8-0fb4-4128-be29-858dbf897551" />

### Conversacion con Chatbot
<img width="1853" height="926" alt="image" src="https://github.com/user-attachments/assets/8a1f2029-909f-477c-8211-785b0ed43f63" />

## Aviso

Herramienta con fines educativos. No reemplaza el criterio de un profesional médico colegiado.
