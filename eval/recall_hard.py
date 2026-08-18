"""Recall@5 sobre terminos medicos exactos y raros.
Si la vectorial ya los encuentra, la busqueda hibrida aporta poco."""
import os, json
from openai import OpenAI
from supabase import create_client

oa = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# (pregunta natural, termino clave que debe aparecer en question_focus)
CASOS = [
    ("What are the symptoms of WAGR syndrome?",                  "wagr"),
    ("Tell me about LCHAD deficiency",                           "lchad"),
    ("What does an MTHFR gene mutation cause?",                  "mthfr"),
    ("How is Crimean-Congo Hemorrhagic Fever transmitted?",      "crimean"),
    ("What is COACH syndrome?",                                  "coach"),
    ("What are the signs of Behcet's disease?",                  "behcet"),
    ("What is Parry-Romberg syndrome?",                          "parry"),
    ("What causes Li-Fraumeni syndrome?",                        "fraumeni"),
    ("What is Hutchinson-Gilford progeria syndrome?",            "progeria"),
    ("Tell me about Dyggve-Melchior-Clausen syndrome",           "dyggve"),
    ("What is colpocephaly?",                                    "colpocephaly"),
    ("What is triploidy?",                                       "triploidy"),
    ("What are the symptoms of Caffey disease?",                 "caffey"),
    ("What is Crigler Najjar syndrome type 2?",                  "crigler"),
    ("How is Wilson disease treated?",                           "wilson"),
    ("What is oculopharyngeal muscular dystrophy?",              "oculopharyngeal"),
    ("What is beta-ketothiolase deficiency?",                    "ketothiolase"),
    ("How is Peyronie's disease treated?",                       "peyronie"),
    ("What is Hermansky-Pudlak syndrome?",                       "hermansky"),
    ("What is pyridoxine-dependent epilepsy?",                   "pyridoxine"),
]

aciertos, fallos = 0, []
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
        print(f"  OK  (pos {hit})  {pregunta[:52]:54} sim={focos[hit-1][1]:.3f}")
    else:
        fallos.append((pregunta, clave, focos))
        print(f"  --  FALLA     {pregunta[:52]:54} top1='{focos[0][0][:32]}'")

print(f"\n{'='*78}")
print(f"recall@5 sobre terminos exactos raros: {aciertos}/{len(CASOS)} = {100*aciertos/len(CASOS):.0f}%")

if fallos:
    print(f"\n--- detalle de los {len(fallos)} fallos ---")
    for pregunta, clave, focos in fallos:
        print(f"\n  {pregunta}   (esperaba '{clave}')")
        for f, s in focos:
            print(f"      {s:.3f}  {f[:60]}")
