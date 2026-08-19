"""Construcción de citas y sustitución de enlaces muertos.

Sin red: `app.citations` solo usa urllib.parse.
"""
import pytest

from app.citations import formatear_cita, fuentes_html, url_de_cita


# ------------------------------------------------- dominios que siguen vivos

@pytest.mark.parametrize("url", [
    "https://ghr.nlm.nih.gov/condition/wilson-disease",       # 301 -> MedlinePlus
    "https://www.cancer.gov/types/testicular/patient/x",
    "http://www.nhlbi.nih.gov/health/health-topics/topics/cf",
    "https://www.nlm.nih.gov/medlineplus/headache.html",      # 301 -> MedlinePlus
])
def test_dominios_vivos_conservan_su_url(url):
    resultado, es_busqueda = url_de_cita(url, "Wilson disease")
    assert resultado == url
    assert es_busqueda is False


# ------------------------------------------------------ dominios con 404

def test_gard_va_al_buscador():
    """GARD es el 34.7% del corpus y devuelve 404 en todas sus URLs de 2017."""
    url, es_busqueda = url_de_cita(
        "https://rarediseases.info.nih.gov/gard/7893/wilson-disease", "Wilson disease"
    )
    assert es_busqueda is True
    assert url == "https://rarediseases.info.nih.gov/diseases?search=Wilson+disease"


def test_niddk_usa_el_parametro_correcto():
    """
    Verificado en navegador: NIDDK ignora `?query=` y solo responde a `?q=`.
    Con el parámetro equivocado devuelve la página de búsqueda vacía.
    """
    url, es_busqueda = url_de_cita(
        "http://www.niddk.nih.gov/health-information/health-topics/x/Pages/facts.aspx",
        "Wilson disease",
    )
    assert es_busqueda is True
    assert url == "https://www.niddk.nih.gov/search?q=Wilson+disease"


def test_ninds_va_al_buscador():
    url, es_busqueda = url_de_cita(
        "http://www.ninds.nih.gov/disorders/wilsons/wilsons.htm", "Wilson Disease"
    )
    assert es_busqueda is True
    assert "search_api_fulltext=Wilson+Disease" in url


def test_nihseniorhealth_sin_buscador_propio():
    """El sitio cerró; su contenido pasó a MedlinePlus, que no tiene ID equivalente."""
    url, es_busqueda = url_de_cita(
        "http://nihseniorhealth.gov/periodontaldisease/toc.html", "Gum Disease"
    )
    assert es_busqueda is True
    assert "medlineplus.gov" in url


# ------------------------------------------------------------- casos borde

def test_sin_question_focus_no_se_puede_buscar():
    """Sin nombre de enfermedad no hay término de búsqueda: se deja la original."""
    url, es_busqueda = url_de_cita("https://rarediseases.info.nih.gov/gard/1/x", "")
    assert url == "https://rarediseases.info.nih.gov/gard/1/x"
    assert es_busqueda is False


def test_url_vacia():
    assert url_de_cita("", "Wilson disease") == ("", False)


def test_dominio_desconocido_se_deja_igual():
    url, es_busqueda = url_de_cita("https://ejemplo.org/algo", "Wilson disease")
    assert url == "https://ejemplo.org/algo"
    assert es_busqueda is False


def test_el_termino_se_codifica():
    """Nombres con espacios, comas o apóstrofes no deben romper la URL."""
    url, _ = url_de_cita(
        "https://rarediseases.info.nih.gov/gard/1/x", "Crigler Najjar syndrome, type 2"
    )
    assert " " not in url
    assert "%2C" in url or "," not in url.split("search=")[1]


# ------------------------------------------------------------ formato final

def test_formato_con_fragmentos():
    cita = formatear_cita({
        "document_source": "GHR",
        "question_focus": "Wilson disease",
        "document_url": "https://ghr.nlm.nih.gov/condition/wilson-disease",
        "chunk_id": 0,
        "n_chunks": 4,
    })
    assert cita.startswith("[GHR] Wilson disease (fragmento 1 de 4)")
    assert "ghr.nlm.nih.gov" in cita
    assert "búsqueda" not in cita


def test_formato_sin_fragmentos_cuando_hay_uno_solo():
    cita = formatear_cita({
        "document_source": "GHR",
        "question_focus": "Wilson disease",
        "document_url": "https://ghr.nlm.nih.gov/condition/wilson-disease",
        "chunk_id": 0,
        "n_chunks": 1,
    })
    assert "fragmento" not in cita


def test_formato_sustituye_la_url_muerta():
    cita = formatear_cita({
        "document_source": "GARD",
        "question_focus": "Wilson disease",
        "document_url": "https://rarediseases.info.nih.gov/gard/7893/wilson-disease",
        "chunk_id": 2,
        "n_chunks": 8,
    })
    assert "(fragmento 3 de 8)" in cita
    assert "diseases?search=" in cita
    assert "/gard/" not in cita


# ------------------------------------------------------ bloque HTML del chat

def test_fuentes_html_vacio_sin_documentos():
    """Sin fuentes no debe quedar un separador suelto colgando de la respuesta."""
    assert fuentes_html([]) == ""


def test_fuentes_html_arma_enlaces():
    html = fuentes_html([{
        "document_source": "GHR",
        "question_focus": "Wilson disease",
        "document_url": "https://ghr.nlm.nih.gov/condition/wilson-disease",
        "chunk_id": 0,
        "n_chunks": 4,
    }])
    assert 'class="fuentes"' in html
    assert 'href="https://ghr.nlm.nih.gov/condition/wilson-disease"' in html
    assert 'target="_blank"' in html
    assert "Wilson disease" in html
    assert "fragmento 1 de 4" in html
    assert "GHR" in html


def test_fuentes_html_omite_fragmento_si_hay_uno_solo():
    html = fuentes_html([{
        "document_source": "GHR",
        "question_focus": "Wilson disease",
        "document_url": "https://ghr.nlm.nih.gov/condition/wilson-disease",
        "chunk_id": 0,
        "n_chunks": 1,
    }])
    assert "fragmento" not in html


def test_fuentes_html_tambien_sustituye_urls_muertas():
    html = fuentes_html([{
        "document_source": "GARD",
        "question_focus": "Wilson disease",
        "document_url": "https://rarediseases.info.nih.gov/gard/7893/wilson-disease",
        "chunk_id": 0,
        "n_chunks": 2,
    }])
    assert "diseases?search=Wilson+disease" in html
    assert "/gard/" not in html
