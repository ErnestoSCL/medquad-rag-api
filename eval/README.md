# Evaluación del retrieval

Scripts usados para decidir, con datos, qué mejoras del pipeline valían la
pena y cuáles no. Se corren contra la base de Supabase ya poblada, con las
variables de entorno del `.env` cargadas.

```bash
python eval/recall_hard.py          # terminos tecnicos exactos y raros
python eval/recall_descriptive.py   # descripciones sin el nombre tecnico
python eval/precision_test.py       # cuanto ruido entra en el top-5
python eval/relative_cutoff.py      # calibracion del corte relativo
```

## Resultados

| Prueba | Resultado |
|---|---|
| `recall_hard` — 20 términos técnicos raros (siglas, nombres propios) | **20/20**, todos en posición 1 |
| `recall_descriptive` — 8 descripciones sin nombre técnico | **7/8** |
| `precision_test` — ruido en el top-5 | 90% antes del corte |
| `relative_cutoff` — corte al 80% del mejor | **98%**, sin perder relevantes |

## Decisiones que salieron de esto

**Búsqueda híbrida (BM25 + vectorial): descartada.** Su ventaja es el término
exacto raro, y ahí el sistema ya da 100% en posición 1. La razón es que el
embedding se calcula sobre `question + chunk_text`, lo que vuelve la búsqueda
pregunta-contra-pregunta y captura siglas y nombres propios directamente.
Agregar `tsvector` + GIN habría costado ~50 MB de base sin ganancia medible.

**Reranking con LLM: descartado.** Sí existía un problema de precisión (10% de
ruido en el top-5), pero el corte relativo lo resuelve igual de bien sin
agregar una llamada por consulta ni ~1 s de latencia.

**Corte relativo: implementado.** Ver `app/rag_chain.py`.

## Nota sobre las métricas

La relevancia se determina por coincidencia de texto contra `question_focus`,
que subestima: en `recall_descriptive`, dos de los tres "fallos" iniciales
recuperaban temas correctos que la clave no matcheaba ("Neonatal progeroid
syndrome" para progeria, "Bleeding Disorders" para hemofilia). Los números
son una cota inferior, no una medición exacta.
