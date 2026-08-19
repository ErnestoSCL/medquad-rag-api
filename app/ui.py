"""Interfaz de chat: barra lateral con conversaciones y chat a pantalla completa.

Se construye con gr.Blocks y no con gr.ChatInterface porque este último no
permite el layout de dos columnas ni manejar varias conversaciones por usuario.

Sobre el CSS: Gradio envuelve todo en varios contenedores con altura automática,
así que para llenar la pantalla hay que forzar la cadena completa desde `html`
hasta el componente. Los selectores por `elem_id` son los únicos estables entre
versiones; las clases internas de Gradio cambian entre releases.
"""
import json

import gradio as gr

from app.memory import (
    cargar_conversacion_completa,
    listar_conversaciones,
)

# Elegidos porque sus fuentes resuelven a la página original, no al buscador
# del sitio. Cerca de la mitad de las URLs de MedQuAD (2017) ya no existen
# —GARD, NIDDK, NINDS y nihseniorhealth se reorganizaron— y para esos dominios
# se enlaza a una búsqueda (ver app/citations.py). Como estos ejemplos son lo
# primero que prueba cualquiera, conviene que muestren el mejor caso.
#
# Se comprobó con peticiones reales que las fuentes de cada uno devuelven 200:
#   sirven      -> NHLBI y GHR, que dejaron redirecciones al migrar
#   se evitaron -> GARD, NINDS, NIDDK y nihseniorhealth, con URLs muertas
#                  (Bell's palsy, Parkinson, diabetes y migraña caían ahí)
EJEMPLOS = [
    "¿Cómo se trata el asma?",
    "¿Qué es la fibrosis quística?",
    "¿Qué es la anemia?",
    "¿Qué causa la hipertensión?",
    "¿Qué es la apnea del sueño?",
    "¿Qué es la osteoporosis?",
]

CSS = """
/* ---- paleta ----------------------------------------------------------
   gr.Blocks en Gradio 6 ya no acepta `theme`, así que el naranja por defecto
   se neutraliza sobrescribiendo las variables del tema. */
:root, .dark {
    --button-primary-background-fill: #18181b;
    --button-primary-background-fill-hover: #27272a;
    --button-primary-border-color: #18181b;
    --button-primary-text-color: #fafafa;
    --color-accent: #52525b;
    --color-accent-soft: rgba(82,82,91,.12);
    --checkbox-background-color-selected: #52525b;
    --checkbox-border-color-selected: #52525b;
    --slider-color: #52525b;
}
.dark {
    --button-primary-background-fill: #fafafa;
    --button-primary-background-fill-hover: #ffffff;
    --button-primary-border-color: #fafafa;
    --button-primary-text-color: #18181b;
}

/* ---- cadena de altura: sin esto el contenido se apelmaza arriba ---- */
html, body { height: 100%; margin: 0; overflow: hidden; }
gradio-app { height: 100vh; display: block; }
.gradio-container {
    height: 100vh !important;
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}
/* `.main.fillable` trae padding 16px 32px y max-width 1280px: sin esto la app
   queda con un marco alrededor y no llega a los bordes de la pantalla. */
.gradio-container > .main,
.main.fillable,
.gradio-container > .main > .wrap,
.gradio-container > .main > .wrap > .contain {
    height: 100% !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}
footer, .built-with, .show-api { display: none !important; }

/* Gradio impone `flex: 1 1 0%` y `min-width: min(320px,100%)` a cada Column,
   por eso las dos se repartían la pantalla mitad y mitad. El ancho del lateral
   se fija con scale=0 + min_width en Python; acá solo se lo acota por arriba y
   se fuerza la altura, que Gradio deja en `auto`. */
#marco { height: 100vh !important; gap: 0 !important; flex-wrap: nowrap !important; }

/* ---------------------------- barra lateral ---------------------------- */
#lateral {
    flex: 0 0 258px !important;
    min-width: 258px !important;
    max-width: 258px !important;
    height: 100vh !important;
    overflow-y: auto;
    padding: 18px 10px !important;
    background: var(--background-fill-secondary);
    border-right: 1px solid var(--border-color-primary);
    gap: 0 !important;
    justify-content: flex-start !important;
}
/* Sin esto, los hijos del lateral se reparten la altura disponible: el botón
   "Nueva conversación" llegaba a medir 572px de alto y quedaban huecos enormes
   entre los elementos. Hay que alcanzar también a los nietos, porque Gradio
   envuelve cada componente en un div propio. */
#lateral > *, #lateral > * > *, #lateral .block {
    flex: 0 0 auto !important;
    width: 100% !important;
    height: auto !important;
    min-height: 0 !important;
    align-self: flex-start !important;
}
#lateral .block, #lateral .form {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}
#marca p { font-size: .88rem; font-weight: 600; margin: 0 0 1px 8px; }
#marca-sub p { font-size: .7rem; opacity: .45; margin: 0 0 14px 8px; }

/* elem_id va en el propio <button> (ver nota en #btn-enviar) */
#btn-nuevo {
    width: 100% !important;
    height: 36px !important;
    max-height: 36px !important;
    justify-content: center;
    font-size: .82rem !important;
    font-weight: 500;
    padding: 0 12px !important;
    border-radius: 8px !important;
    background: transparent !important;
    border: 1px solid var(--border-color-primary) !important;
}
#btn-nuevo:hover { background: var(--background-fill-primary) !important; }
#rotulo p {
    font-size: .67rem; text-transform: uppercase; letter-spacing: .07em;
    opacity: .45; margin: 22px 0 4px 8px;
}

/* El historial es un gr.Radio, pero tiene que leerse como una lista de
   conversaciones: se ocultan los círculos y cada opción se estira a todo el
   ancho como un ítem clicable. */
#historial, #historial fieldset, #historial .wrap, #historial > div {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    gap: 1px !important;
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
}
#historial input[type="radio"] { display: none !important; }
#historial label {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    border-radius: 7px !important;
    padding: 7px 9px !important;
    margin: 0 !important;
    font-size: .81rem !important;
    font-weight: 400 !important;
    line-height: 1.35;
    width: 100% !important;
    max-width: 100% !important;
    min-height: 0 !important;
    height: auto !important;
    flex: 0 0 auto !important;
    display: block !important;
    cursor: pointer;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    color: var(--body-text-color);
}
#historial label > span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; }
#historial label:hover { background: var(--background-fill-primary) !important; }
#historial label.selected, #historial label:has(input:checked) {
    background: var(--background-fill-primary) !important;
    font-weight: 500 !important;
}

/* ------------------------------ panel chat ----------------------------- */
/* flex-direction: column es imprescindible. Sin él, Gradio deja el Column en
   `row` y la cabecera, el chat y la barra de entrada se acomodan una al lado de
   la otra: el contenido llegaba a 2542px de ancho en una caja de 1022px, con
   los mensajes corridos a la derecha y la entrada fuera de la pantalla. */
#principal {
    height: 100vh !important;
    padding: 0 !important;
    gap: 0 !important;
    min-width: 0 !important;
    flex: 1 1 auto !important;
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    overflow: hidden;
}

/* cabecera: da aire arriba y evita que el chat quede pegado al borde */
#cabecera {
    flex: 0 0 auto !important;
    max-width: 760px;
    width: 100%;
    margin: 0 auto;
    padding: 34px 20px 18px 20px !important;
    gap: 0 !important;
    border: none !important;
}
#titulo h1, #titulo p {
    font-size: 1.28rem !important;
    font-weight: 600;
    letter-spacing: -.015em;
    margin: 0 0 4px 0 !important;
}
#subtitulo p {
    font-size: .82rem !important;
    opacity: .5;
    margin: 0 !important;
    line-height: 1.45;
}

#conversacion {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: auto !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    /* `hidden`, no `auto`: el scroll real lo hace `.bubble-wrap`, el contenedor
       interno del Chatbot. Con overflow auto acá aparecían DOS barras
       superpuestas mientras se generaba la respuesta. Igual hace falta acotar
       la altura (flex + min-height: 0), o el chat empuja la barra de entrada
       fuera de la pantalla. */
    overflow: hidden !important;
}
/* el wrapper que Gradio pone alrededor del Chatbot también debe estirarse */
#conversacion > div { height: 100% !important; }
#conversacion > .wrap { padding: 4px 0 8px 0; }

/* El chat se centra en la pantalla, no se pega al borde del panel.
   `width: 100%` es necesario: sin él la fila se encoge al tamaño de su
   contenido y el margin auto la centra, con lo que las preguntas cortas
   quedaban flotando en el medio en vez de alinearse a la derecha. */
#conversacion .message-row,
#conversacion .message-wrap > div {
    width: 100% !important;
    max-width: 760px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: 20px; padding-right: 20px;
}
#conversacion .message-row.user-row { justify-content: flex-end !important; }
#conversacion .message-row.user-row > * { margin-left: auto !important; }

/* burbujas: la pregunta con fondo tenue, la respuesta sin caja */
#conversacion .bot-row .message,
#conversacion .message.bot {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding-left: 0 !important;
}
/* La pregunta se alinea al borde derecho del área de lectura. Gradio la deja
   centrada por defecto (medido: iba de x=651 a 887 dentro de un área de
   409..1129), lo que la hacía parecer flotando en el medio. */
#conversacion .user-row .message,
#conversacion .message.user {
    background: var(--background-fill-secondary) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 16px !important;
    padding: 10px 15px !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    max-width: 80% !important;
    width: fit-content !important;
}
#conversacion .flex-wrap.role:has(.message.user) { justify-content: flex-end !important; }

/* jerarquía tipográfica dentro de las respuestas */
#conversacion .message.user p { font-size: .9rem !important; line-height: 1.5; margin: 0; }
#conversacion .message.bot p { font-size: .96rem !important; line-height: 1.68; }
#conversacion .message.bot li { font-size: .96rem !important; line-height: 1.6; }
#conversacion .message.bot strong { font-weight: 600; }
#conversacion .message.bot em { font-size: .8rem; opacity: .6; font-style: normal; }

/* ------------------------------- entrada ------------------------------- */
#zona-inferior {
    flex: 0 0 auto !important;
    max-width: 760px;
    margin: 0 auto;
    width: 100%;
    padding: 6px 20px 16px 20px !important;
    gap: 8px !important;
    border: none !important;
}
#barra { gap: 8px !important; align-items: flex-end; }
#entrada textarea {
    border-radius: 14px !important;
    padding: 12px 14px !important;
    font-size: .92rem !important;
    resize: none;
    border: 1px solid var(--border-color-primary) !important;
    background: var(--background-fill-primary) !important;
}
#entrada textarea:focus { outline: none; border-color: var(--body-text-color-subdued) !important; }
/* Dos trampas acá:
   1. Gradio pone el elem_id en el propio <button>, no en un contenedor, así
      que `#btn-enviar button` no selecciona nada.
   2. Redefinir --button-primary-background-fill en :root no alcanza: Gradio la
      vuelve a declarar en un ancestro del botón (en :root vale lo que uno pone,
      pero leída desde el botón sigue siendo #ea580c). Hay que redefinirla en el
      propio elemento. */
#btn-enviar, button.primary {
    --button-primary-background-fill: #18181b !important;
    --button-primary-background-fill-hover: #27272a !important;
    --button-primary-border-color: #18181b !important;
    --button-primary-text-color: #fafafa !important;
    border-radius: 12px !important;
    min-width: 84px; height: 42px;
    font-size: .85rem !important; font-weight: 500;
    background: #18181b !important;
    border: 1px solid #18181b !important;
    color: #fafafa !important;
}
#btn-enviar:hover, button.primary:hover { background: #27272a !important; }
.dark #btn-enviar, .dark button.primary {
    --button-primary-background-fill: #fafafa !important;
    --button-primary-text-color: #18181b !important;
    background: #fafafa !important;
    border-color: #fafafa !important;
    color: #18181b !important;
}
.dark #btn-enviar:hover, .dark button.primary:hover { background: #ffffff !important; }
#aviso p { text-align: center; font-size: .7rem; opacity: .42; margin: 6px 0 0 0; }

/* ------------------------------ ejemplos ------------------------------- */
#chips {
    gap: 6px !important;
    flex-wrap: wrap !important;
    border: none !important;
    background: transparent !important;
}
#chips > * { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
#chips button {
    font-size: .76rem !important;
    font-weight: 400 !important;
    padding: 5px 11px !important;
    border-radius: 999px !important;
    min-width: 0 !important;
    width: auto !important;
    background: var(--background-fill-secondary) !important;
    border: 1px solid var(--border-color-primary) !important;
    white-space: nowrap;
    opacity: .82;
}
#chips button:hover { opacity: 1; }

/* ------------------------------ scrollbars ----------------------------- */
/* Gris discreta en lugar de la del sistema. `.bubble-wrap` es el contenedor
   interno del Chatbot, que es quien scrollea de verdad. */
.bubble-wrap, #lateral, #entrada textarea {
    scrollbar-width: thin;
    scrollbar-color: #52525b transparent;
}
.bubble-wrap::-webkit-scrollbar,
#lateral::-webkit-scrollbar,
#entrada textarea::-webkit-scrollbar { width: 8px; height: 8px; }

.bubble-wrap::-webkit-scrollbar-track,
#lateral::-webkit-scrollbar-track,
#entrada textarea::-webkit-scrollbar-track { background: transparent; }

/* El borde transparente con background-clip deja aire a los lados del pulgar,
   para que no toque el borde del panel. */
.bubble-wrap::-webkit-scrollbar-thumb,
#lateral::-webkit-scrollbar-thumb,
#entrada textarea::-webkit-scrollbar-thumb {
    background: #52525b;
    border-radius: 4px;
    border: 2px solid transparent;
    background-clip: content-box;
}
.bubble-wrap::-webkit-scrollbar-thumb:hover,
#lateral::-webkit-scrollbar-thumb:hover,
#entrada textarea::-webkit-scrollbar-thumb:hover {
    background: #71717a;
    background-clip: content-box;
}

/* ---------------------------- fuentes citadas -------------------------- */
.fuentes {
    margin-top: 16px; padding-top: 10px;
    border-top: 1px solid var(--border-color-primary);
}
.fuentes-titulo {
    text-transform: uppercase; letter-spacing: .07em;
    font-size: .65rem; opacity: .45; margin-bottom: 7px;
}
.fuente { font-size: .78rem; line-height: 1.75; }
.fuente a { text-decoration: none; font-weight: 500; border-bottom: 1px solid transparent; }
.fuente a:hover { border-bottom-color: currentColor; }
.fuente-meta { opacity: .5; }
"""


def construir(responder, iniciar_sesion, refrescar_lista):
    """
    `responder(mensaje, historial_ui, user_id, conversation_id)`
        -> (entrada, historial, conversation_id)
    `iniciar_sesion(user_id)` -> (user_id, conversation_id, opciones)
    `refrescar_lista(user_id, conversation_id)` -> gr.update para la barra lateral
    """
    with gr.Blocks(title="Asistente Médico RAG",
                   fill_height=True, analytics_enabled=False) as demo:

        # Se usa gr.State y el localStorage se maneja a mano.
        # gr.BrowserState lanza "Unexpected end of JSON input" al releer su
        # valor (con default None y con ""), y ese error rompe el ciclo de
        # eventos de Gradio: la interfaz deja de responder a cualquier clic.
        usuario = gr.State("")
        conversacion = gr.State("")

        # Puente para traer el id de usuario desde el navegador. Hace falta
        # porque el `js` de un evento NO alimenta los inputs de su `fn`: el
        # valor devuelto se ignoraba y el servidor recibía "", generando un
        # usuario nuevo en cada recarga (28 usuarios para 32 conversaciones) y
        # perdiendo el historial. Con un componente en `outputs`, el JS sí
        # escribe, y un segundo evento encadenado lo lee ya actualizado.
        uid_puente = gr.Textbox(value="", visible=False)

        # scale=0 + min_width fija el ancho del lateral: con el default
        # (scale=1) Gradio reparte el espacio en partes iguales.
        with gr.Row(elem_id="marco", equal_height=True):
            with gr.Column(scale=0, min_width=258, elem_id="lateral"):
                gr.Markdown("Asistente Médico", elem_id="marca")
                gr.Markdown("Corpus MedQuAD · NIH", elem_id="marca-sub")
                btn_nuevo = gr.Button("Nueva conversación", size="sm",
                                      variant="secondary", elem_id="btn-nuevo")
                gr.Markdown("Historial", elem_id="rotulo")
                historial_ui = gr.Radio(choices=[], show_label=False,
                                        container=False, elem_id="historial")

            with gr.Column(scale=1, min_width=0, elem_id="principal"):
                with gr.Column(elem_id="cabecera"):
                    gr.Markdown("Asistente Médico RAG", elem_id="titulo")
                    gr.Markdown(
                        "Consultas sobre síntomas, tratamientos y enfermedades, "
                        "con respuestas basadas en el corpus MedQuAD",
                        elem_id="subtitulo",
                    )

                # En Gradio 6 los botones de la burbuja se controlan con
                # `buttons` (no existen show_copy_button/show_share_button), y
                # sanitize_html=False es necesario para que se rendericen las
                # fuentes en HTML.
                chat = gr.Chatbot(
                    elem_id="conversacion", show_label=False, height="100%",
                    buttons=[], avatar_images=(None, None),
                    sanitize_html=False, layout="bubble",
                    group_consecutive_messages=False,
                    placeholder=(
                        "<div style='opacity:.35;font-size:.9rem;text-align:center'>"
                        "Preguntá sobre síntomas, tratamientos o enfermedades</div>"
                    ),
                )

                with gr.Column(elem_id="zona-inferior"):
                    # Los ejemplos van como chips en una fila que se ajusta al
                    # ancho, no como barras de ancho completo dentro de un
                    # acordeón (que Gradio pinta como un desplegable).
                    with gr.Row(elem_id="chips"):
                        chips = [gr.Button(e, size="sm", variant="secondary",
                                           scale=0, min_width=0)
                                 for e in EJEMPLOS]

                    with gr.Row(elem_id="barra"):
                        entrada = gr.Textbox(
                            placeholder="Escribí tu pregunta médica",
                            show_label=False, container=False, lines=1,
                            max_lines=6, scale=8, elem_id="entrada", autofocus=True,
                        )
                        btn_enviar = gr.Button("Enviar", variant="primary",
                                               scale=0, elem_id="btn-enviar")

                    gr.Markdown(
                        "Información con fines educativos. No sustituye el criterio "
                        "de un profesional médico colegiado.",
                        elem_id="aviso",
                    )

        # ------------------------------------------------------- eventos

        def _al_cargar(uid):
            uid, cid, opciones = iniciar_sesion(uid)
            # Si el usuario ya tenía conversaciones se reabre la última con sus
            # mensajes; si no, el chat arranca vacío.
            mensajes = cargar_conversacion_completa(cid) if cid else []
            return uid, cid, gr.update(choices=opciones, value=cid or None), mensajes

        # Este JS se ejecuta al cargar y escribe el id de usuario en
        # `uid_puente`. Hace dos cosas que Gradio 6 no resuelve por sí solo:
        #
        #  1. Inyecta el CSS. `gr.Blocks(css=...)` y `head=` se descartan en
        #     silencio, y el `head` de gr.HTML llega escapado al documento.
        #  2. Lee o crea el id de usuario en localStorage, reemplazando a
        #     gr.BrowserState (ver comentario en su declaración).
        js_inicio = (
            "() => {"
            "  if (!document.getElementById('css-medquad')) {"
            "    const s = document.createElement('style');"
            "    s.id = 'css-medquad';"
            f"    s.textContent = {json.dumps(CSS)};"
            "    document.head.appendChild(s);"
            "  }"
            # El naranja del botón primario resiste a la cascada: Gradio
            # redefine --button-primary-background-fill en un ancestro y
            # recalcula el estilo después de montar. Se fija por estilo inline,
            # que es lo único que sobrevive.
            "  const pintar = () => {"
            "    const oscuro = document.body.classList.contains('dark')"
            "      || document.querySelector('.dark') !== null;"
            "    const fondo = oscuro ? '#fafafa' : '#18181b';"
            "    const texto = oscuro ? '#18181b' : '#fafafa';"
            "    document.querySelectorAll('button.primary').forEach(b => {"
            "      b.style.setProperty('background', fondo, 'important');"
            "      b.style.setProperty('border-color', fondo, 'important');"
            "      b.style.setProperty('color', texto, 'important');"
            "    });"
            "  };"
            "  pintar();"
            "  setTimeout(pintar, 300);"
            "  new MutationObserver(pintar).observe(document.body,"
            "    { childList: true, subtree: true });"
            # Clave nueva ('medquad_user'): bajo 'medquad_uid' quedó el valor
            # que dejó gr.BrowserState, que guarda cifrado y no es un UUID.
            # El chequeo de formato descarta cualquier residuo de ese tipo.
            "  let uid = null;"
            "  try { uid = localStorage.getItem('medquad_user'); } catch (e) {}"
            "  const valido = uid && uid.length > 20 && !uid.includes(':');"
            "  if (!valido) {"
            "    uid = (crypto.randomUUID ? crypto.randomUUID()"
            "          : 'u-' + Date.now() + '-' + Math.random().toString(36).slice(2));"
            "    try { localStorage.setItem('medquad_user', uid); } catch (e) {}"
            "  }"
            "  return uid;"
            "}"
        )

        # Dos pasos encadenados: el JS escribe el uid en `uid_puente` (con
        # fn=None su retorno sí va a `outputs`), y recién entonces _al_cargar lo
        # lee. Pasarle `js` directamente a _al_cargar no sirve: Gradio ignora el
        # valor devuelto y la función recibe el input vacío.
        demo.load(None, inputs=None, outputs=[uid_puente], js=js_inicio).then(
            _al_cargar,
            inputs=[uid_puente],
            outputs=[usuario, conversacion, historial_ui, chat],
        )

        def _nueva(uid):
            """
            Limpia el chat sin registrar nada en la base: la conversación se
            crea con el primer mensaje (ver _responder). Si se creara acá, cada
            clic dejaría una entrada vacía en la lista que sobrevive a la
            recarga.
            """
            return "", gr.update(choices=listar_conversaciones(uid), value=None), []

        btn_nuevo.click(_nueva, inputs=[usuario],
                        outputs=[conversacion, historial_ui, chat])

        def _abrir(elegida, actual):
            if not elegida or elegida == actual:
                return gr.skip(), gr.skip()
            return elegida, cargar_conversacion_completa(elegida)

        historial_ui.change(_abrir, inputs=[historial_ui, conversacion],
                            outputs=[conversacion, chat])

        entradas = [entrada, chat, usuario, conversacion]
        # La barra lateral NO va acá: se refresca en un evento encadenado. Si
        # fuera output de `responder`, Gradio pintaría su indicador
        # "processing | 4.7s" sobre el historial mientras se genera la
        # respuesta, que es donde menos sentido tiene.
        salidas = [entrada, chat, conversacion]

        entrada.submit(responder, inputs=entradas, outputs=salidas).then(
            refrescar_lista, inputs=[usuario, conversacion], outputs=[historial_ui]
        )
        btn_enviar.click(responder, inputs=entradas, outputs=salidas).then(
            refrescar_lista, inputs=[usuario, conversacion], outputs=[historial_ui]
        )

        # Cada chip llama directo a `responder` con su propio texto. El patrón
        # de rellenar el textbox y encadenar con .then() no funciona: el
        # segundo paso lee el valor viejo del componente.
        def _handler_de_chip(texto):
            def _enviar(historial_ui, uid, cid):
                return responder(texto, historial_ui, uid, cid)
            return _enviar

        # El texto sale de EJEMPLOS, no de `chip.value`: ese atributo no
        # devuelve el rótulo en Gradio 6 y el handler recibía None.
        for chip, texto_ejemplo in zip(chips, EJEMPLOS):
            chip.click(
                _handler_de_chip(texto_ejemplo),
                inputs=[chat, usuario, conversacion],
                outputs=salidas,
            ).then(
                refrescar_lista, inputs=[usuario, conversacion], outputs=[historial_ui]
            )

    return demo
