import re

# ============================================================
# Guardrail 1 — Protección de PII (regex)
# ============================================================

# Los usuarios preguntan en español, pero la consulta interna se traduce al
# inglés antes de buscar (ver rag_chain.reformular). Por eso los guardrails
# cubren AMBOS idiomas en vez de reemplazar el inglés por español: los patrones
# en español protegen la entrada del usuario, los de inglés siguen cubriendo la
# API y el texto que circula internamente.
PII_PATTERNS = {
    'email'    : re.compile(r'[\w\.-]+@[\w\.-]+\.\w+'),
    'telefono' : re.compile(r'(\+?\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}'),
    # El nombre admite acentos y ñ: "Soy José Martínez", "me llamo Iñaki".
    'nombre'   : re.compile(
        r'\b(?:my name is|i am|i\'m|soy|me llamo|mi nombre es)\s+'
        r'([A-ZÁÉÍÓÚÑ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+)?)',
        flags=re.IGNORECASE
    ),
}


def mask_pii(text):
    """
    Detecta y enmascara PII en el texto de entrada.
    Devuelve (texto_enmascarado, dict_con_detecciones).
    """
    masked = text
    detections = {}

    for label, pattern in PII_PATTERNS.items():
        matches = pattern.findall(masked)
        if matches:
            detections[label] = len(matches) if isinstance(matches[0], str) or label != 'nombre' else len(matches)
            if label == 'nombre':
                masked = pattern.sub(lambda m: m.group(0).split()[0] + ' [NOMBRE_OCULTO]', masked)
            else:
                masked = pattern.sub(f'[{label.upper()}_OCULTO]', masked)

    return masked, detections


def strip_pii(text):
    """
    Igual que mask_pii, pero ELIMINA el PII en vez de sustituirlo por un
    placeholder. Es la versión que se usa en el pipeline (búsqueda y LLM).

    Por qué existe: los placeholders tipo `[NOMBRE_OCULTO]` son tokens sin
    significado que perjudican las dos etapas.

      · Búsqueda — diluyen el embedding. Medido sobre "My name is John Smith,
        what causes my headaches?", la mejor similitud cae de 0.538 (sin PII)
        a 0.483 (con placeholder) y los chunks relevantes en el top-5 bajan de
        3 a 1 — peor, incluso, que no haber enmascarado nada.
      · Generación — con el placeholder en la pregunta, gpt-4o-mini se abstiene
        ("I don't know") aun teniendo la respuesta en el contexto.

    No se relaja la protección: eliminar es al menos tan seguro como sustituir,
    y el PII no llega ni a la base vectorial ni al LLM. La versión enmascarada
    de mask_pii sigue disponible en `apply_guardrails` para logs y auditoría,
    junto con el detalle de qué se detectó.
    """
    stripped = text
    for pattern in PII_PATTERNS.values():
        stripped = pattern.sub('', stripped)

    # Limpia lo que deja la eliminación: espacios dobles, espacio antes de
    # puntuación, y puntuación huérfana al inicio ("...Smith, what causes"
    # queda como ", what causes" -> "what causes").
    stripped = re.sub(r'\s{2,}', ' ', stripped)
    stripped = re.sub(r'\s+([,.;:!?])', r'\1', stripped)
    stripped = re.sub(r'^[\s,;:.]+', '', stripped)
    return stripped.strip()


# ============================================================
# Guardrail 2 — Defensa contra prompt injection
# ============================================================

INJECTION_PATTERNS = [
    # --- inglés ---
    r'ignore (all |the )?(previous|above|prior) instructions',
    r'disregard (the |your )?(system prompt|instructions|rules)',
    r'you are now',
    r'forget (everything|all) (you|that)',
    r'act as (if you|a different|an unrestricted)',
    r'new instructions\s*:',
    r'system\s*:\s*you (must|should|will)',
    r'override (your |the )?(rules|guidelines|instructions)',
    r'pretend (you are|to be)',
    r'reveal your (system )?prompt',
    # --- español ---
    # Se aceptan variantes con y sin tilde: quien intenta una inyección no
    # necesariamente escribe con ortografía correcta.
    r'ignora (todas )?(las )?(instrucciones|reglas|indicaciones)',
    r'olvida (todo|lo anterior|las instrucciones|tus instrucciones)',
    r'haz caso omiso (de|a) (las |tus )?(instrucciones|reglas)',
    r'no sigas (las |tus )?(instrucciones|reglas)',
    r'ahora eres',
    r'a partir de ahora (eres|ser[áa]s|act[úu]a)',
    r'act[úu]a como (si|un|una)',
    r'finge (que |ser )',
    r'compórtate como',
    r'nuevas instrucciones\s*:',
    r'sistema\s*:\s*(debes|tienes que)',
    r'(revela|mu[ée]strame|dime) (tu|el) (prompt|system prompt|mensaje de sistema)',
    r'anula (tus|las) (reglas|instrucciones|directrices)',
]

INJECTION_REGEX = re.compile('|'.join(INJECTION_PATTERNS), flags=re.IGNORECASE)


def detect_prompt_injection(text):
    """
    Devuelve True si el texto contiene un patrón típico de prompt injection.
    A diferencia de PII, aquí no se sanitiza: se bloquea la pregunta completa,
    porque "limpiar" texto adversarial es mucho menos confiable que rechazarlo.
    """
    return bool(INJECTION_REGEX.search(text))


# ============================================================
# Guardrail 3 — Filtro de sesgos y toxicidad (lista de términos)
# ============================================================

# Lista breve e ilustrativa de términos ofensivos/discriminatorios. En un
# sistema productivo se ampliaría y se mantendría en un archivo de
# configuración versionado, o se delegaría a un clasificador.
#
# Cubre los dos idiomas: el usuario escribe en español, pero la respuesta del
# LLM y la query interna viajan en inglés, y este mismo filtro se aplica a la
# respuesta generada (ver main.py).
TOXIC_TERMS = {
    # --- inglés ---
    'idiot', 'stupid', 'retard', 'retarded', 'moron',
    'kill yourself', 'kys',
    'subhuman', 'inferior race',
    # --- español ---
    'idiota', 'estúpido', 'estupido', 'estúpida', 'estupida',
    'imbécil', 'imbecil', 'retrasado', 'retrasada', 'subnormal',
    'tarado', 'tarada', 'mongólico', 'mongolico',
    'matate', 'mátate', 'suicídate', 'suicidate',
    'infrahumano', 'raza inferior',
}

# Términos que exigen palabra completa para no producir falsos positivos con
# vocabulario clínico legítimo. Sin esto, "tarado" dispararía dentro de
# "retardado" y "retard" dentro de "retardation", que aparece en el corpus
# médico en inglés como término técnico.
TOXIC_WORD_BOUNDED = {
    'retard', 'retarded', 'tarado', 'tarada', 'retrasado', 'retrasada',
}


def contains_toxicity(text):
    """
    Búsqueda de términos ofensivos/discriminatorios sobre el texto en
    minúsculas. Devuelve (bool, lista_de_términos_encontrados).

    Los términos de TOXIC_WORD_BOUNDED se buscan como palabra completa, porque
    aparecen como subcadena dentro de vocabulario médico legítimo.
    """
    text_lower = text.lower()
    found = []
    for term in TOXIC_TERMS:
        if term in TOXIC_WORD_BOUNDED:
            if re.search(rf'\b{re.escape(term)}\b', text_lower):
                found.append(term)
        elif term in text_lower:
            found.append(term)
    return (len(found) > 0, sorted(found))


# ============================================================
# Guardrail 4 — Cumplimiento clínico (disclaimer + mensaje de
# información insuficiente) — aplicado a la RESPUESTA generada
# ============================================================

CLINICAL_DISCLAIMER = (
    "*Nota: esta información tiene fines educativos y no sustituye el criterio "
    "de un profesional médico colegiado.*"
)

INSUFFICIENT_INFO_MSG = (
    "No hay información suficiente en el material de referencia. Se recomienda "
    "acudir a un especialista en el tema."
)


# Formas de abstención que hay que reconocer. El LLM responde en español, pero
# el marcador interno de "sin contexto" que devuelve rag_chain.answer_question
# sigue siendo "I don't know", y el modelo puede recaer en inglés, así que se
# cubren ambos idiomas.
NO_ANSWER_MARKERS = (
    "i don't know",
    "i do not know",
    "no lo sé",
    "no lo se",
    "no tengo información",
    "no tengo informacion",
    "no puedo responder",
)


def apply_clinical_guardrails(answer):
    """
    Envuelve la respuesta cruda del LLM con los guardrails clínicos:
      - Si el modelo no encontró la respuesta en el contexto, la reemplaza por
        el mensaje estándar de información insuficiente.
      - Agrega siempre el disclaimer obligatorio al final.
    """
    answer_clean = answer.strip()
    lower = answer_clean.lower()

    no_answer = not answer_clean or any(m in lower for m in NO_ANSWER_MARKERS)
    body = INSUFFICIENT_INFO_MSG if no_answer else answer_clean

    # Si el modelo ya escribió el disclaimer, no se agrega otro. Pasaba cuando
    # el historial conservaba respuestas con la nota: el LLM veía ese patrón en
    # los turnos previos y lo reproducía, quedando duplicado. La causa se
    # corrigió guardando la respuesta cruda (ver main.py), pero esto queda como
    # segunda barrera.
    if CLINICAL_DISCLAIMER in body or "fines educativos y no sustituye" in body:
        return body

    return f"{body}\n\n{CLINICAL_DISCLAIMER}"


# ============================================================
# Guardrails combinados — apply_guardrails()
# ============================================================

def apply_guardrails(question):
    """
    Aplica los guardrails 1-3 sobre la pregunta de entrada, en orden:
      1. Prompt injection -> bloqueo si se detecta (no se sanitiza).
      2. Toxicidad/sesgos -> bloqueo si se detecta.
      3. PII -> enmascaramiento (no bloquea, solo limpia antes de seguir).
    (El guardrail 4, clínico, se aplica sobre la respuesta, en main.py).

    Devuelve un dict:
      {
        'allowed'         : bool,
        'reason'          : str o None,
        'safe_question'   : la pregunta enmascarada — solo para logs/auditoría,
        'search_question' : la pregunta sin PII — la que usa el pipeline, tanto
                            para buscar como para el LLM (ver strip_pii),
        'pii_detections'  : dict de detecciones de PII (vacío si no hubo),
      }
    """
    if detect_prompt_injection(question):
        return {
            'allowed': False,
            'reason': 'prompt_injection_detected',
            'safe_question': None,
            'search_question': None,
            'pii_detections': {},
        }

    is_toxic, toxic_terms = contains_toxicity(question)
    if is_toxic:
        return {
            'allowed': False,
            'reason': f'toxicity_detected: {toxic_terms}',
            'safe_question': None,
            'search_question': None,
            'pii_detections': {},
        }

    safe_question, pii_found = mask_pii(question)

    return {
        'allowed': True,
        'reason': None,
        'safe_question': safe_question,
        'search_question': strip_pii(question),
        'pii_detections': pii_found,
    }
