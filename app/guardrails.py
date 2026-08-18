import re

# ============================================================
# Guardrail 1 — Protección de PII (regex)
# ============================================================

PII_PATTERNS = {
    'email'    : re.compile(r'[\w\.-]+@[\w\.-]+\.\w+'),
    'telefono' : re.compile(r'(\+?\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}'),
    'nombre'   : re.compile(
        r'\b(?:my name is|i am|i\'m|soy|me llamo)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)',
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


# ============================================================
# Guardrail 2 — Defensa contra prompt injection
# ============================================================

INJECTION_PATTERNS = [
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

# Lista breve e ilustrativa de términos ofensivos/discriminatorios en inglés
# (el corpus y el LLM operan en inglés). En un sistema productivo esta lista
# se ampliaría y se mantendría en un archivo de configuración versionado.
TOXIC_TERMS = {
    'idiot', 'stupid', 'retard', 'retarded', 'moron',
    'kill yourself', 'kys',
    # términos discriminatorios genéricos de ejemplo — lista no exhaustiva
    'subhuman', 'inferior race',
}


def contains_toxicity(text):
    """
    Búsqueda simple de términos ofensivos/discriminatorios, sobre el texto
    en minúsculas. Devuelve (bool, lista_de_términos_encontrados).
    """
    text_lower = text.lower()
    found = [term for term in TOXIC_TERMS if term in text_lower]
    return (len(found) > 0, found)


# ============================================================
# Guardrail 4 — Cumplimiento clínico (disclaimer + mensaje de
# información insuficiente) — aplicado a la RESPUESTA generada
# ============================================================

CLINICAL_DISCLAIMER = (
    "*Note: This information is for educational purposes only and does not "
    "replace the judgment of a licensed physician.*"
)

INSUFFICIENT_INFO_MSG = (
    "Insufficient information in the reference material. Immediate referral "
    "to a specialist for clinical examination is recommended."
)


def apply_clinical_guardrails(answer):
    """
    Envuelve la respuesta cruda del LLM con los guardrails clínicos:
      - Si el modelo no encontró la respuesta en el contexto (responde
        "I don't know" o algo vacío), la reemplaza por el mensaje estándar
        de información insuficiente.
      - Agrega siempre el disclaimer obligatorio al final.
    """
    answer_clean = answer.strip()

    no_answer = (
        not answer_clean
        or "i don't know" in answer_clean.lower()
        or "i do not know" in answer_clean.lower()
    )
    body = INSUFFICIENT_INFO_MSG if no_answer else answer_clean

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
        'allowed'        : bool,
        'reason'         : str o None,
        'safe_question'  : la pregunta a usar en el pipeline (enmascarada si aplica),
        'pii_detections' : dict de detecciones de PII (vacío si no hubo),
      }
    """
    if detect_prompt_injection(question):
        return {
            'allowed': False,
            'reason': 'prompt_injection_detected',
            'safe_question': None,
            'pii_detections': {},
        }

    is_toxic, toxic_terms = contains_toxicity(question)
    if is_toxic:
        return {
            'allowed': False,
            'reason': f'toxicity_detected: {toxic_terms}',
            'safe_question': None,
            'pii_detections': {},
        }

    safe_question, pii_found = mask_pii(question)

    return {
        'allowed': True,
        'reason': None,
        'safe_question': safe_question,
        'pii_detections': pii_found,
    }
