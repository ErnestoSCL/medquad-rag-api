# Pruebas

```
tests/
├── conftest.py                    fixtures compartidas
├── unit/                          sin red, sin servidor, sin credenciales
│   ├── test_guardrails.py         PII, inyección, toxicidad, clínico (es/en)
│   └── test_reranker_parse.py     parseo de la respuesta del reranker
└── integration/                   requieren el contenedor levantado
    ├── test_ask_endpoint.py       POST /ask de extremo a extremo
    └── test_memoria.py            persistencia, ventana, aislamiento
```

## Correr

```bash
# unitarios: solo necesitan pytest
pytest tests/unit -q

# integración: requieren el contenedor y las credenciales del .env
docker run -d --name medquad-test -p 7860:7860 --env-file .env medquad-rag
pytest tests/integration -q

# todo
pytest -q
```

Los de integración se **saltan solos** si el servidor no responde: no tenerlo
levantado no significa que el código esté roto.

Para apuntar a otra instancia (por ejemplo el despliegue de Render):

```bash
MEDQUAD_URL=https://tuapp.onrender.com pytest tests/integration -q
```

## Por qué esta separación

El entorno local no tiene `langchain` ni `gradio`, y no conviene instalarlos
solo para testear. `app/guardrails.py` únicamente importa `re`, así que se
puede probar directo; todo lo que depende del modelo o del vector store se
verifica por HTTP contra el contenedor, que sí los tiene.

Por el mismo motivo, `parse_rerank_response` está separada de `_rerank` en
`app/rag_chain.py`: es la parte frágil —el LLM devuelve texto libre que hay que
convertir en índices— y así se puede cubrir sin red ni modelo.

## Qué cubren los tests de integración

Cada uno gasta 2-3 llamadas al LLM, así que la suite tarda algunos minutos y
cuesta unos centavos. Los casos elegidos son los que en su momento revelaron
fallos reales:

| Test | Regresión que previene |
|---|---|
| `test_primera_persona_se_responde` | el generador leía "mis dolores de cabeza" como pedido de diagnóstico personal y se abstenía |
| `test_la_consulta_de_busqueda_es_impersonal` | el reformulador conservaba el posesivo al traducir y el reranker descartaba los 20 pasajes |
| `test_pii_se_elimina_y_la_pregunta_se_responde` | el placeholder `[NOMBRE_OCULTO]` degradaba la búsqueda y hacía abstenerse al modelo |
| `test_fuera_del_corpus_se_abstiene_sin_citar` | sin contexto, el modelo respondía de memoria y sin fuentes |
| `test_tema_con_un_solo_chunk_no_rellena_citas` | se mostraban 5 fuentes cuando solo una era del tema |
| `test_inyeccion_en_espanol_se_bloquea` | los patrones estaban solo en inglés: en español no bloqueaban nada |
| `test_vocabulario_clinico_no_dispara_toxicidad` | "retardation" y "retraso mental" activaban el filtro de toxicidad |
| `test_busca_en_ingles_aunque_se_pregunte_en_espanol` | buscar en español recupera 3.0/5 chunks contra 4.9/5 traduciendo |

## Limpieza

Los tests de memoria usan `session_id` con prefijo `test-` y borran su
historial al terminar. Si alguna corrida queda a medias:

```sql
delete from chat_history where session_id like 'test-%';
```
