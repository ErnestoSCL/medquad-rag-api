"""Construcción de las citas que se muestran al usuario.

El problema que resuelve: MedQuAD se recopiló alrededor de 2017 y guarda la URL
del documento original. Desde entonces varios sitios del NIH se reorganizaron, y
cerca del 49% de esas URLs devuelven 404.

Medido sobre una muestra del corpus, por dominio:

    dominio                     % corpus   estado
    rarediseases.info.nih.gov     34.7%    404  (GARD reorganizó su catálogo)
    ghr.nlm.nih.gov               23.7%    OK   (301 -> MedlinePlus Genetics)
    www.cancer.gov                13.1%    OK
    www.nhlbi.nih.gov              8.5%    OK
    www.niddk.nih.gov              7.1%    404  (cambió de estructura)
    www.nlm.nih.gov                4.6%    OK   (301 -> MedlinePlus)
    www.ninds.nih.gov              3.8%    404  (rediseño del sitio)
    nihseniorhealth.gov            2.5%    el sitio ya no existe
    www.cdc.gov                    2.0%    parcial

Los que siguen funcionando son los que dejaron redirecciones 301 al migrar.

Se descartó reescribir las rutas: para GARD, cambiar `/gard/ID/slug` por
`/diseases/ID/slug` funcionaba en el caso de Wilson disease, pero al validarlo
sobre 10 URLs solo 2 respondían. GARD no movió las páginas, consolidó su
catálogo, y muchas entradas de 2017 ya no existen bajo ningún ID.

La solución es enlazar al buscador del sitio usando el nombre de la enfermedad
(`question_focus`), que sí lleva al contenido aunque no a la página exacta.
"""
from urllib.parse import quote_plus

# Plantillas de búsqueda verificadas en navegador (los parámetros importan:
# NIDDK ignora `?query=` y solo responde a `?q=`).
BUSCADORES = {
    "rarediseases.info.nih.gov": "https://rarediseases.info.nih.gov/diseases?search={}",
    "www.niddk.nih.gov":         "https://www.niddk.nih.gov/search?q={}",
    "www.ninds.nih.gov":         "https://www.ninds.nih.gov/search?search_api_fulltext={}",
    # nihseniorhealth.gov cerró y su contenido se absorbió en MedlinePlus.
    "nihseniorhealth.gov":       "https://medlineplus.gov/spanish/healthtopics.html",
}


def _dominio(url: str) -> str:
    """Extrae el host sin depender de urlparse para URLs mal formadas."""
    if not url:
        return ""
    sin_esquema = url.split("://", 1)[-1]
    return sin_esquema.split("/", 1)[0].lower()


def url_de_cita(document_url: str, question_focus: str) -> tuple[str, bool]:
    """
    Devuelve (url_a_mostrar, es_busqueda).

    Si el dominio está entre los que rompieron sus enlaces, se devuelve una
    búsqueda por `question_focus` en ese mismo sitio. Si no, la URL original.

    `es_busqueda` permite avisar en la interfaz que el enlace lleva a una
    búsqueda y no al documento exacto — mejor eso que dar a entender que es la
    fuente literal.
    """
    dominio = _dominio(document_url)
    plantilla = BUSCADORES.get(dominio)

    if not plantilla or not question_focus:
        return document_url, False

    # nihseniorhealth no tiene buscador propio: su plantilla no lleva {}
    if "{}" not in plantilla:
        return plantilla, True

    return plantilla.format(quote_plus(question_focus)), True


def formatear_cita(metadata: dict) -> str:
    """
    Arma la línea de cita que ve el usuario, en markdown.

    Ejemplo:
        [GARD] Wilson disease (fragmento 1 de 8) — https://...  (búsqueda)
    """
    fuente = metadata.get("document_source")
    foco = metadata.get("question_focus")
    url_original = metadata.get("document_url") or ""

    n_chunks = metadata.get("n_chunks", 1) or 1
    if n_chunks > 1:
        frag = f" (fragmento {metadata['chunk_id'] + 1} de {n_chunks})"
    else:
        frag = ""

    url, _ = url_de_cita(url_original, foco)
    return f"[{fuente}] {foco}{frag} — {url}"


def fuentes_html(metadatos: list[dict]) -> str:
    """
    Bloque de fuentes para el chat, en HTML, con el nombre como enlace y el
    resto como metadatos atenuados. Cadena vacía si no hay fuentes, para que la
    respuesta no arrastre un separador suelto.

    Las clases (`fuentes`, `fuente`, `fuente-meta`) están definidas en el CSS
    de app/ui.py.
    """
    if not metadatos:
        return ""

    filas = []
    for indice, m in enumerate(metadatos, start=1):
        foco = m.get("question_focus") or "Fuente"
        fuente = m.get("document_source") or ""
        url, _ = url_de_cita(m.get("document_url") or "", foco)

        n_chunks = m.get("n_chunks", 1) or 1
        partes = [fuente] if fuente else []
        if n_chunks > 1:
            partes.append(f"fragmento {m['chunk_id'] + 1} de {n_chunks}")
        meta = " · ".join(partes)

        descripcion = foco + (" · " + meta if meta else "")
        etiqueta = ' aria-label="' + descripcion + '" title="' + descripcion + '"'
        marcador = '<span class="fuente-indice">' + str(indice) + '</span>'
        if url:
            filas.append(
                '<a class="fuente" href="' + url + '" target="_blank" '
                'rel="noopener"' + etiqueta + '>' + marcador + '</a>'
            )
        else:
            filas.append('<span class="fuente"' + etiqueta + '>' + marcador + '</span>')

    return (
        '\n\n<div class="fuentes">'
        '<div class="fuentes-titulo">Fuentes</div>'
        '<div class="fuentes-list">'
        + "".join(filas)
        + '</div>'
        + "</div>"
    )
