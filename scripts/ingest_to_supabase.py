import pandas as pd
from openai import OpenAI
from supabase import create_client
import os

# text-embedding-3-small produce 1536 dimensiones por defecto, pero admite
# truncarlo con `dimensions` (Matryoshka). Se usan 512: los 38,127 vectores
# pasan de ~223 MB a ~78 MB —y su índice ivfflat de ~300 MB a ~80 MB—, lo que
# mantiene el proyecto dentro del free tier de Supabase (0.5 GB) a cambio de
# ~1% de calidad en MTEB. Debe coincidir con vector(512) en sql/schema.sql y
# con OpenAIEmbeddings(dimensions=...) en app/rag_chain.py.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512

# Permite ingestar a una tabla intermedia durante una migración, sin tocar la
# tabla en uso: INGEST_TABLE=documents_512 python scripts/ingest_to_supabase.py
TABLE = os.environ.get("INGEST_TABLE", "documents")

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

df = pd.read_parquet("df_chunks_B_512.parquet")
print(f"Ingestando {len(df)} chunks a `{TABLE}` con {EMBEDDING_DIMENSIONS} dimensiones")

BATCH = 100
for start in range(0, len(df), BATCH):
    batch = df.iloc[start:start + BATCH]

    # Se embeddea question + chunk_text combinados: el usuario pregunta, no
    # recita pasajes de respuesta, así que comparar pregunta-contra-pregunta
    # (más el chunk) da un matching más directo que comparar solo contra el
    # texto de la respuesta. `content` (lo que ve el LLM) NO cambia — sigue
    # siendo solo chunk_text, esto solo afecta qué tan bien se *encuentra*.
    texts_to_embed = [
        f"{row.question}\n{row.chunk_text}" for row in batch.itertuples()
    ]

    resp = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts_to_embed,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    vectors = [d.embedding for d in resp.data]

    rows = [
        {
            "content": row.chunk_text,
            "metadata": {
                "question": row.question,
                "document_source": row.document_source,
                "document_url": row.document_url,
                "question_focus": row.question_focus,
                "document_id": row.document_id,
                "chunk_id": int(row.chunk_id),
                "n_chunks": int(row.n_chunks),
            },
            "embedding": vec,
        }
        for row, vec in zip(batch.itertuples(), vectors)
    ]
    supabase.table(TABLE).insert(rows).execute()
    print(f"Insertados {start + len(batch)}/{len(df)}")

print("Ingesta completa.")
