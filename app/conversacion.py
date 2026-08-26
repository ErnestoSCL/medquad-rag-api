"""Mensajes que no son consultas médicas: saludos y preguntas sobre el asistente.

Sin esto, "hola, ¿en qué me podés ayudar?" entra al pipeline RAG, no recupera
nada del corpus y termina en "No hay información suficiente en el material de
referencia" — una respuesta desconcertante para lo que en realidad es el primer
mensaje de cualquier usuario.

Se resuelve por patrones y no con el LLM por tres razones: es determinista, no
suma latencia a un pipeline que ya hace tres llamadas, y evita que el modelo
invente capacidades que el sistema no tiene.

El criterio para no pisar consultas legítimas: si al quitar el saludo queda
contenido con sustancia, el mensaje sigue al RAG. Así "hola, ¿qué es la
diabetes?" se responde como pregunta médica, no como saludo.
"""
import re

BIENVENIDA = (
    "Soy un asistente de consulta médica. Respondo preguntas sobre síntomas, "
    "tratamientos, causas y diagnósticos, usando únicamente el corpus **MedQuAD** "
    "de los Institutos Nacionales de Salud de EE. UU., y cito las fuentes de cada "
    "respuesta.\n\n"
    "Podés preguntarme cosas como:\n\n"
    "- ¿Cuáles son los síntomas de la parálisis de Bell?\n"
    "- ¿Cómo se trata el asma?\n"
    "- ¿Qué causa los dolores de cabeza?\n\n"
    "Recuerdo las últimas interacciones, así que podés repreguntar sobre lo "
    "mismo: *«¿y cómo se trata?»*.\n\n"
    "No tengo información fuera de ese corpus, y cuando no la tengo lo digo en "
    "vez de improvisar."
)

# Saludos y cortesías. Se recortan del mensaje para ver si queda algo más.
SALUDOS = re.compile(
    r'\b(hola+|buenas?(\s+(d[íi]as|tardes|noches))?|buen\s+d[íi]a|qu[ée]\s+tal|'
    r'hey|holi|saludos|hi|hello|gracias|muchas\s+gracias|ad[íi]os|chau|'
    r'buenos\s+d[íi]as|c[óo]mo\s+est[áa]s|qu[ée]\s+hac[ée]s)\b',
    flags=re.IGNORECASE,
)

# Preguntas sobre el asistente en sí: qué hace, cómo funciona, quién es.
#
# El verbo va como comodín (`\S*`) en vez de enumerar conjugaciones: la gente
# escribe rápido y con errores. El caso que motivó esto fue "hola en que me
# pedes ayudar" — con una lista cerrada de verbos, ese typo se escapaba y el
# usuario recibía "No hay información suficiente".
# Cada alternativa arranca con \b. Sin ese límite, `qu[ée]\s+haces?` —pensado
# para "¿qué haces?"— disparaba dentro de "porque hace unos días" y una consulta
# médica larga terminaba recibiendo el mensaje de bienvenida.
META = re.compile(
    r'(\ben\s+qu[ée]\s+(me\s+)?(\S+\s+)?ayudar|'
    r'\b(pod[ée]s|puedes|pedes|podes)\s+ayudar|'
    r'\b(qu[ée]|c[óo]mo)\s+(pod[ée]s|puedes|pedes|podes|sab[ée]s|haces|funcionas?)\b|'
    r'\bpara\s+qu[ée]\s+(sirves?|est[áa]s)\b|'
    r'\bqui[ée]n\s+(sos|eres)\b|\bqu[ée]\s+(sos|eres)\b|'
    r'\bc[óo]mo\s+(te\s+)?(uso|utilizo|funciona)\b|'
    r'\bqu[ée]\s+(tipo\s+de\s+)?(preguntas?|consultas?)\s+(puedo|te\s+puedo)\b|'
    r'\b(ay[úu]dame|necesito\s+ayuda)\b|'
    r'\bde\s+d[óo]nde\s+(sacas|obtienes)\s+(la\s+)?informaci[óo]n|'
    r'\bqu[ée]\s+es\s+(esto|este\s+chat|medquad)\b|'
    r'\bwhat\s+can\s+you\s+do\b|\bwho\s+are\s+you\b|\bhow\s+do\s+you\s+work\b)',
    flags=re.IGNORECASE,
)

# Palabras vacías que quedan tras recortar el saludo y no aportan sustancia.
_RESIDUO = re.compile(r'^[\s,.!¡?¿:;\-–—]*$')

# Longitud mínima, en caracteres, para considerar que lo que sobra del mensaje
# es una consulta real y no el resto de un saludo.
_MIN_SUSTANCIA = 12


def es_conversacional(mensaje: str) -> bool:
    """
    True si el mensaje es solo un saludo o una pregunta sobre el asistente, y
    por lo tanto debe responderse sin pasar por el RAG.

    Devuelve False en cuanto el mensaje tiene una consulta real, aunque venga
    acompañada de un saludo.
    """
    if not mensaje or not mensaje.strip():
        return True

    texto = mensaje.strip()

    # Una meta-pregunta se atiende siempre, aunque el mensaje sea largo.
    if META.search(texto):
        return True

    # Para los saludos: se recortan y se mira si queda algo con sustancia.
    resto = SALUDOS.sub(" ", texto)
    resto = re.sub(r'\s{2,}', " ", resto).strip()

    # Si no había ningún saludo que recortar, el mensaje es una consulta, por
    # corta que sea. Antes el umbral de longitud se aplicaba igual y "asma",
    # "diabetes" o "anemia" —consultas perfectamente válidas— recibían el
    # mensaje de bienvenida en lugar de una respuesta.
    if resto == texto:
        return False

    if _RESIDUO.match(resto):
        return True

    return len(resto) < _MIN_SUSTANCIA


def respuesta_conversacional() -> str:
    return BIENVENIDA
