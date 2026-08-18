"""Caso dificil de verdad: el usuario NO sabe el nombre tecnico y describe.
Aqui BM25 tampoco ayudaria (no hay termino exacto que machear)."""
import os, json
from openai import OpenAI
from supabase import create_client

oa = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

CASOS = [
    ("What is the disease where children age very rapidly?",          "progeria"),
    ("Why is half of my face weak and drooping suddenly?",            "bell"),
    ("I have a burning pain on one side of my head with tearing eye", "sunct"),
    ("What causes copper to build up in the liver and brain?",        "wilson"),
    ("My child has a white pupil in one eye, what could it be?",      "retinoblastom"),
    ("What illness makes you unable to stop bleeding after a cut?",   "hemophilia"),
    ("Why do my hands shake when I try to grab something?",           "tremor"),
    ("What disease causes thick mucus in the lungs and pancreas?",    "cystic fibrosis"),
]

aciertos = 0
for pregunta, clave in CASOS:
    emb = oa.embeddings.create(model="text-embedding-3-small",
                               input=pregunta, dimensions=512).data[0].embedding
    r = sb.rpc("match_documents", {"query_embedding": emb, "match_count": 5}).execute()
    focos = []
    for d in r.data:
        m = d["metadata"] if isinstance(d["metadata"], dict) else json.loads(d["metadata"])
        focos.append((m.get("question_focus") or "", d["similarity"]))

    hit = next((i for i, (f, _) in enumerate(focos, 1) if clave in f.lower()), None)
    if hit:
        aciertos += 1
        print(f"  OK (pos {hit})  {pregunta[:50]:52} -> {focos[hit-1][0][:34]}")
    else:
        print(f"  -- FALLA     {pregunta[:50]:52} -> {[f[:22] for f, _ in focos[:3]]}")

print(f"\nrecall@5 con descripciones (sin nombre tecnico): {aciertos}/{len(CASOS)} = {100*aciertos/len(CASOS):.0f}%")
print("\nNOTA: la clave es coincidencia de texto y subestima. Revisar los fallos a mano:")
print("      'Neonatal progeroid syndrome' y 'Bleeding Disorders' son respuestas validas")
print("      aunque no matcheen 'progeria' ni 'hemophilia'.")
