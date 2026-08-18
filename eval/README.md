# Evaluación del retrieval

Scripts usados para decidir, con datos, qué mejoras del pipeline valían la
pena y cuáles no. Se corren contra la base de Supabase ya poblada, con las
variables de entorno del `.env` cargadas.

```bash
python eval/recall_hard.py          # terminos tecnicos exactos y raros
python eval/recall_descriptive.py   # descripciones sin el nombre tecnico
python eval/precision_test.py       # cuanto ruido entra en el top-5
python eval/relative_cutoff.py      # calibracion del corte relativo
python eval/pool_test.py            # hay relevantes en las posiciones 6-20?
```

## Resultados

| Prueba | Resultado |
|---|---|
| `recall_hard` — 20 términos técnicos raros (siglas, nombres propios) | **20/20**, todos en posición 1 |
| `recall_descriptive` — 8 descripciones sin nombre técnico | **7/8** |
| `precision_test` — ruido en el top-5 | 90% antes del corte |
| `relative_cutoff` — corte al 80% del mejor | **98%**, sin perder relevantes |
| `pool_test` — relevantes en posiciones 6-20 | solo **1 de 12** consultas ganaría algo |

## Decisiones

**Búsqueda híbrida (BM25 + vectorial): descartada.** Su ventaja es el término
exacto raro, y ahí el sistema ya da 100% en posición 1. La razón es que el
embedding se calcula sobre `question + chunk_text`, lo que vuelve la búsqueda
pregunta-contra-pregunta y captura siglas y nombres propios directamente.
Agregar `tsvector` + GIN habría costado ~50 MB de base sin ganancia medible.

**Corte relativo: implementado.** Resuelve el ruido del top-5 sin costo.

**Reranking con LLM: implementado, con la salvedad de que su aporte es
marginal.** `pool_test` muestra que ampliar el pool de 5 a 20 candidatos solo
ayudaría en 1 de 12 consultas: donde el corpus tiene material de sobra el top-5
ya venía completo, y donde tenía huecos es porque no existen más chunks del
tema. Lo que sí aporta es juicio semántico sobre la relevancia, que afina las
citas mostradas al usuario (Bell's palsy pasa de 5 fuentes a 3). Costo medido:
+0.4 s por consulta.

## Un hallazgo del reranking: alucinación con contexto vacío

Al activarlo apareció un problema que no tenía que ver con el ranking. Cuando
el reranker descartaba todos los pasajes, el contexto quedaba vacío y
`gpt-4o-mini` **respondía de memoria** en lugar de abstenerse: "What is Caffey
disease?" devolvía una respuesta correcta y **sin ninguna fuente**.

El system prompt dice `If the context does not contain the answer, say 'I don't
know'`, pero eso es una instrucción, no una garantía. La abstención ahora se
fuerza en código: si no quedan documentos, `answer_question` devuelve
"I don't know" sin llegar a llamar al LLM.

## Nota sobre las métricas

La relevancia se determina por coincidencia de texto contra `question_focus`,
que subestima: en `recall_descriptive`, dos de los tres "fallos" iniciales
recuperaban temas correctos que la clave no matcheaba ("Neonatal progeroid
syndrome" para progeria, "Bleeding Disorders" para hemofilia). Los números
son una cota inferior, no una medición exacta.
