import pandas as pd
from openai import OpenAI
from supabase import create_client
import os

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

df = pd.read_parquet("df_chunks_B_512.parquet")

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

    resp = openai_client.embeddings.create(model="text-embedding-3-small", input=texts_to_embed)
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
    supabase.table("documents").insert(rows).execute()
    print(f"Insertados {start + len(batch)}/{len(df)}")

print("Ingesta completa.")
