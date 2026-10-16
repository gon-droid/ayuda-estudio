import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
import pypdf
import json
import time
from pydantic import BaseModel

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA Y ESTILOS
# ==========================================
st.set_page_config(
    page_title="StudyAI - Plataforma de Estudio Inteligente",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .flashcard-box {
        background-color: #1e293b;
        color: #f8fafc;
        border: 2px solid #3b82f6;
        border-radius: 16px;
        padding: 40px 25px;
        text-align: center;
        min-height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .flashcard-badge {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #93c5fd;
        margin-bottom: 12px;
        font-weight: 700;
    }
    .flashcard-content {
        font-size: 1.35rem;
        font-weight: 500;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# ESQUEMAS PYDANTIC PARA SALIDAS JSON
# ==========================================
class OpcionesTest(BaseModel):
    A: str
    B: str
    C: str
    D: str

class PreguntaTest(BaseModel):
    pregunta: str
    opciones: OpcionesTest
    respuesta_correcta: str
    justificaciones: OpcionesTest

class CuestionarioSchema(BaseModel):
    preguntas: list[PreguntaTest]

class FlashcardItem(BaseModel):
    anverso: str
    reverso: str

class ListaFlashcardsSchema(BaseModel):
    tarjetas: list[FlashcardItem]

class TrampaExamen(BaseModel):
    concepto: str
    error_comun: str
    como_evitarlo: str

class ResumenYTrampasSchema(BaseModel):
    puntos_clave: list[str]
    trampas_examen: list[TrampaExamen]

# ==========================================
# INICIALIZACIÓN DEL ESTADO DE SESIÓN
# ==========================================
def init_session_state():
    defaults = {
        "study_text": "",
        "quiz_data": None,
        "user_answers": {},
        "flashcards_data": None,
        "fc_index": 0,
        "fc_flipped": False,
        "fc_status": {},
        "summary_data": None,
        "chat_history": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==========================================
# FUNCIONES AUXILIARES Y TOLERANCIA A FALLOS (503)
# ==========================================
def get_genai_client(api_key: str):
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Error al inicializar el cliente de Google GenAI: {e}")
        return None

def call_gemini_with_retry(client, primary_model: str, contents, config: types.GenerateContentConfig):
    """
    Realiza la llamada a la API con reintentos automáticos y cambio de modelo (fallback)
    si el servidor de Google devuelve un error 503 (alta demanda) o 429.
    """
    fallback_models = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-3.8-pro"]
    # Poner el modelo elegido por el usuario en primer lugar sin duplicar
    models_to_try = [primary_model] + [m for m in fallback_models if m != primary_model]

    last_error = None
    for model_name in models_to_try:
        for attempt in range(2):  # Hasta 2 intentos por modelo
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
            except APIError as e:
                last_error = e
                err_str = str(e).upper()
                # Si es sobrecarga temporal (503) o límite de velocidad (429), esperamos y reintentamos
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    time.sleep(2 * (attempt + 1))
                    continue
                # Si el modelo no está disponible en esa cuenta (404), pasamos al siguiente modelo de la lista
                elif "404" in err_str or "NOT_FOUND" in err_str:
                    break
                else:
                    raise e
    raise last_error

def extract_text_from_file(uploaded_file) -> str:
    try:
        if uploaded_file.type == "application/pdf":
            reader = pypdf.PdfReader(uploaded_file)
            extracted_pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            text = "\n\n".join(extracted_pages)
            if not text.strip():
                st.warning("⚠️ El PDF parece no contener texto seleccionable.")
            return text
        elif uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
    return ""

# ==========================================
# BARRA LATERAL: CONFIGURACIÓN E INGESTA
# ==========================================
with st.sidebar:
    st.title("⚙️ Configuración")
    
    default_api_key = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            default_api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    if default_api_key:
        st.success("🔑 API Key activa y conectada")
        use_custom_key = st.checkbox("Usar mi propia API Key personal")
        raw_key = st.text_input("Introduce tu Gemini API Key:", type="password") if use_custom_key else default_api_key
    else:
        st.markdown("👉 **[Consigue tu API Key gratis aquí](https://aistudio.google.com/app/apikey)**")
        raw_key = st.text_input("Introduce tu Gemini API Key:", type="password")

    api_key = raw_key.strip().strip('"').strip("'") if raw_key else ""

    MODEL_ID = st.selectbox(
        "Modelo de IA preferido:",
        options=["gemini-3.8-flash", "gemini-3.8-pro", "gemini-2.0-flash"],
        index=0,
        help="Si el modelo elegido experimenta alta demanda (Error 503), la app reintentará automáticamente."
    )

    st.divider()
    st.subheader("📚 Entrada de Material")

    input_mode = st.radio(
        "Método de carga:",
        options=["Subir archivo (PDF / TXT)", "Pegar texto directamente"]
    )

    if input_mode == "Subir archivo (PDF / TXT)":
        uploaded_file = st.file_uploader("Selecciona tus apuntes", type=["pdf", "txt"])
        if uploaded_file is not None:
            if st.button("📥 Procesar y Cargar Archivo", use_container_width=True, type="primary"):
                with st.spinner("Extrayendo texto del documento..."):
                    extracted = extract_text_from_file(uploaded_file)
                    if extracted.strip():
                        st.session_state.study_text = extracted
                        st.toast("¡Material cargado con éxito!", icon="✅")
    else:
        manual_text = st.text_area(
            "Pega aquí tus apuntes o temario:",
            height=220,
            placeholder="Introduce el contenido teórico que deseas estudiar..."
        )
        if st.button("📥 Guardar Texto", use_container_width=True, type="primary"):
            if manual_text.strip():
                st.session_state.study_text = manual_text
                st.toast("¡Texto guardado con éxito!", icon="✅")
            else:
                st.warning("El campo de texto está vacío.")

    if st.session_state.study_text:
        st.divider()
        char_count = len(st.session_state.study_text)
        word_count = len(st.session_state.study_text.split())
        st.caption(f"📊 **Material activo:** {word_count:,} palabras ({char_count:,} caracteres)")
        
        with st.expander("👁️ Previsualizar temario cargado"):
            st.text(st.session_state.study_text[:1500] + ("..." if char_count > 1500 else ""))
            
        if st.button("🗑️ Limpiar todo el material", use_container_width=True):
            for key in ["study_text", "quiz_data", "user_answers", "flashcards_data", "summary_data", "chat_history"]:
                st.session_state[key] = "" if key == "study_text" else ([] if key == "chat_history" else ({} if "answers" in key else None))
            st.rerun()

# ==========================================
# ÁREA PRINCIPAL Y VALIDACIÓN PREVIA
# ==========================================
st.title("🎓 StudyAI: Asistente de Estudio Interactivo")
st.markdown("Transforma tus apuntes en cuestionarios justificados, tarjetas de memoria, resúmenes estratégicos y resuelve dudas con tu tutor de IA.")

if not api_key:
    st.info("👈 Por favor, introduce tu **Gemini API Key** en el panel lateral para comenzar.")
    st.stop()

if not st.session_state.study_text:
    st.info("👈 Sube un archivo **PDF/TXT** o pega tus apuntes en el panel lateral para activar los módulos de estudio.")
    st.stop()

client = get_genai_client(api_key)
if not client:
    st.stop()

# ==========================================
# NAVEGACIÓN POR PESTAÑAS (MÓDULOS)
# ==========================================
tab_quiz, tab_flashcards, tab_summary, tab_tutor = st.tabs([
    "📝 1. Cuestionarios Tipo Test",
    "🗂️ 2. Flashcards / Memoria",
    "📌 3. Resumen y Preguntas Trampa",
    "💬 4. Tutor Interactivo (Chat)"
])

# ------------------------------------------
# PESTAÑA 1: CUESTIONARIOS TIPO TEST
# ------------------------------------------
with tab_quiz:
    st.header("Cuestionario de Autoevaluación")
    
    col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 1, 1])
    with col_cfg1:
        num_questions = st.selectbox("Número de preguntas:", options=[5, 10, 15], index=0)
    with col_cfg2:
        difficulty = st.selectbox("Nivel de dificultad:", options=["Fácil", "Intermedio", "Difícil (Nivel Examen)"])
    with col_cfg3:
        st.write("")
        st.write("")
        btn_generate_quiz = st.button("🎲 Generar Nuevo Test", type="primary", use_container_width=True)

    if btn_generate_quiz:
        with st.spinner("Diseñando preguntas y justificaciones pedagógicas..."):
            prompt_quiz = f"""
            Actúa como un profesor universitario riguroso. A partir del temario proporcionado, genera un examen tipo test de exactamente {num_questions} preguntas con nivel de dificultad '{difficulty}'.
            
            Requisitos obligatorios:
            1. Cada pregunta debe tener 4 opciones: A, B, C y D. Solo una es verdadera.
            2. El campo 'respuesta_correcta' debe contener únicamente la letra en mayúscula ("A", "B", "C" o "D").
            3. En 'justificaciones', explica detalladamente paso a paso por qué la opción correcta es verdadera y el motivo exacto por el cual cada una de las otras tres opciones es falsa o actúa como distractor.
            
            TEMARIO DE REFERENCIA:
            {st.session_state.study_text[:65000]}
            """
            try:
                response = call_gemini_with_retry(
                    client=client,
                    primary_model=MODEL_ID,
                    contents=prompt_quiz,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=CuestionarioSchema,
                        temperature=0.3
                    )
                )
                parsed_quiz = json.loads(response.text)
                st.session_state.quiz_data = parsed_quiz.get("preguntas", [])
                st.session_state.user_answers = {}
                st.rerun()
            except APIError as e:
                st.error(f"Los servidores de Google están saturados en este instante. Por favor, pulsa el botón de nuevo en unos segundos. Detalle: {e}")
            except Exception as e:
                st.error(f"Error inesperado al generar el cuestionario: {e}")

    if st.session_state.quiz_data:
        questions = st.session_state.quiz_data
        total_q = len(questions)
        answered_q = len(st.session_state.user_answers)
        
        correct_count = sum(
            1 for idx, ans in st.session_state.user_answers.items()
            if ans == questions[idx]["respuesta_correcta"].strip().upper()
        )

        st.markdown("---")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Progreso", f"{answered_q} / {total_q} respondidas")
        m_col2.metric("Aciertos actuales", f"{correct_count} ✅")
        nota_sobre_10 = (correct_count / total_q) * 10 if total_q > 0 else 0
        m_col3.metric("Nota Provisional (sobre 10)", f"{nota_sobre_10:.2f}")
        st.progress(answered_q / total_q if total_q > 0 else 0)

        if answered_q == total_q and total_q > 0:
            if nota_sobre_10 >= 5.0:
                st.balloons()
                st.success(f"🏆 **¡Cuestionario completado!** Has obtenido un **{nota_sobre_10:.2f}/10** ({correct_count} de {total_q} aciertos).")
            else:
                st.warning(f"📚 **Cuestionario finalizado.** Has obtenido un **{nota_sobre_10:.2f}/10** ({correct_count} de {total_q} aciertos). Revisa las justificaciones inferiores para reforzar los conceptos.")

        st.markdown("---")

        for i, q in enumerate(questions):
            with st.container(border=True):
                st.subheader(f"Pregunta {i + 1}")
                st.markdown(f"**{q['pregunta']}**")

                opciones = q["opciones"]
                correct_letter = q["respuesta_correcta"].strip().upper()
                user_choice = st.session_state.user_answers.get(i)

                for letra in ["A", "B", "C", "D"]:
                    texto_opcion = opciones.get(letra, "")
                    col_btn, col_txt = st.columns([1, 11])
                    
                    with col_btn:
                        is_selected = (user_choice == letra)
                        btn_type = "primary" if is_selected else "secondary"
                        if st.button(
                            letra,
                            key=f"q_{i}_opt_{letra}",
                            disabled=(user_choice is not None),
                            type=btn_type,
                            use_container_width=True
                        ):
                            st.session_state.user_answers[i] = letra
                            st.rerun()
                    
                    with col_txt:
                        if user_choice is not None:
                            if letra == correct_letter:
                                st.markdown(f"✅ **{texto_opcion}** *(Respuesta Correcta)*")
                            elif letra == user_choice and user_choice != correct_letter:
                                st.markdown(f"❌ ~~{texto_opcion}~~ *(Tu elección)*")
                            else:
                                st.markdown(f"{texto_opcion}")
                        else:
                            st.markdown(f"{texto_opcion}")

                if user_choice is not None:
                    if user_choice == correct_letter:
                        st.success(f"¡Correcto! La opción **{correct_letter}** es la adecuada.")
                    else:
                        st.error(f"Incorrecto. Marcaste la opción **{user_choice}**, pero la respuesta correcta es la **{correct_letter}**.")

                    with st.expander("📖 Ver justificación teórica paso a paso de todas las opciones", expanded=True):
                        justificaciones = q["justificaciones"]
                        for letra in ["A", "B", "C", "D"]:
                            es_valida = (letra == correct_letter)
                            icono = "🟢" if es_valida else "🔴"
                            estado_txt = "CORRECTA" if es_valida else "INCORRECTA"
                            st.markdown(f"**{icono} Opción {letra} ({estado_txt}):** {justificaciones.get(letra, '')}")

        if answered_q > 0:
            if st.button("🔄 Reiniciar mis respuestas de este test", use_container_width=True):
                st.session_state.user_answers = {}
                st.rerun()

# ------------------------------------------
# PESTAÑA 2: FLASHCARDS / TARJETAS DE MEMORIA
# ------------------------------------------
with tab_flashcards:
    st.header("Tarjetas de Memoria (Active Recall)")

    col_fc_cfg1, col_fc_cfg2 = st.columns([2, 1])
    with col_fc_cfg1:
        num_cards = st.slider("Cantidad de tarjetas a generar:", min_value=5, max_value=20, value=10, step=1)
    with col_fc_cfg2:
        st.write("")
        btn_generate_fc = st.button("🗂️ Generar Flashcards", type="primary", use_container_width=True)

    if btn_generate_fc:
        with st.spinner("Extrayendo conceptos clave y definiciones..."):
            prompt_fc = f"""
            A partir del temario proporcionado, genera exactamente {num_cards} tarjetas de memoria (flashcards) diseñadas para practicar Active Recall.
            - En 'anverso': coloca un concepto clave, término técnico, fórmula o pregunta directa concisa.
            - En 'reverso': coloca la definición clara, explicación exacta o respuesta directa sin rodeos.
            
            TEMARIO DE REFERENCIA:
            {st.session_state.study_text[:65000]}
            """
            try:
                response = call_gemini_with_retry(
                    client=client,
                    primary_model=MODEL_ID,
                    contents=prompt_fc,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ListaFlashcardsSchema,
                        temperature=0.3
                    )
                )
                parsed_fc = json.loads(response.text)
                st.session_state.flashcards_data = parsed_fc.get("tarjetas", [])
                st.session_state.fc_index = 0
                st.session_state.fc_flipped = False
                st.session_state.fc_status = {}
                st.rerun()
            except APIError as e:
                st.error(f"Alta demanda temporal en Google AI. Inténtalo de nuevo en unos segundos: {e}")
            except Exception as e:
                st.error(f"Error al generar las tarjetas: {e}")

    if st.session_state.flashcards_data:
        cards = st.session_state.flashcards_data
        total_cards = len(cards)
        idx = st.session_state.fc_index

        learned_count = sum(1 for status in st.session_state.fc_status.values() if status == "Aprendida")
        review_count = sum(1 for status in st.session_state.fc_status.values() if status == "Repasar")
        pending_count = total_cards - learned_count - review_count

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("✅ Aprendidas", learned_count)
        kpi2.metric("🔄 Para Repasar", review_count)
        kpi3.metric("⏳ Pendientes", pending_count)

        st.markdown("---")

        current_card = cards[idx]
        current_status = st.session_state.fc_status.get(idx, "Sin clasificar")
        status_badge = "✅ Aprendida" if current_status == "Aprendida" else ("🔄 Repasar" if current_status == "Repasar" else "⚪ Sin clasificar")

        st.markdown(f"**Tarjeta {idx + 1} de {total_cards}** — *Estado actual: {status_badge}*")

        cara_titulo = "REVERSO (Definición / Explicación)" if st.session_state.fc_flipped else "ANVERSO (Concepto / Pregunta)"
        contenido_mostrar = current_card["reverso"] if st.session_state.fc_flipped else current_card["anverso"]
        borde_color = "#10b981" if st.session_state.fc_flipped else "#3b82f6"

        st.markdown(f"""
        <div class="flashcard-box" style="border-color: {borde_color};">
            <div class="flashcard-badge">{cara_titulo}</div>
            <div class="flashcard-content">{contenido_mostrar}</div>
        </div>
        """, unsafe_allow_html=True)

        c_nav1, c_nav2, c_nav3 = st.columns([1, 2, 1])
        with c_nav1:
            if st.button("⬅️ Anterior", disabled=(idx == 0), use_container_width=True):
                st.session_state.fc_index -= 1
                st.session_state.fc_flipped = False
                st.rerun()
        with c_nav2:
            label_flip = "🔄 Volver al Concepto (Anverso)" if st.session_state.fc_flipped else "🔄 Voltear Tarjeta (Ver Respuesta)"
            if st.button(label_flip, type="primary", use_container_width=True):
                st.session_state.fc_flipped = not st.session_state.fc_flipped
                st.rerun()
        with c_nav3:
            if st.button("Siguiente ➡️", disabled=(idx == total_cards - 1), use_container_width=True):
                st.session_state.fc_index += 1
                st.session_state.fc_flipped = False
                st.rerun()

        st.write("")
        c_mark1, c_mark2 = st.columns(2)
        with c_mark1:
            if st.button("✅ Marcar como 'Aprendida'", use_container_width=True):
                st.session_state.fc_status[idx] = "Aprendida"
                if idx < total_cards - 1:
                    st.session_state.fc_index += 1
                    st.session_state.fc_flipped = False
                st.rerun()
        with c_mark2:
            if st.button("📌 Marcar para 'Repasar'", use_container_width=True):
                st.session_state.fc_status[idx] = "Repasar"
                if idx < total_cards - 1:
                    st.session_state.fc_index += 1
                    st.session_state.fc_flipped = False
                st.rerun()

        with st.expander("📋 Ver listado completo de tarjetas y su estado"):
            for c_i, card in enumerate(cards):
                st_i = st.session_state.fc_status.get(c_i, "Pendiente")
                icon_i = "✅" if st_i == "Aprendida" else ("🔄" if st_i == "Repasar" else "⏳")
                st.markdown(f"**{c_i + 1}. {icon_i} {st_i}**  \n*R:* {card['reverso']}")
                st.divider()

# ------------------------------------------
# PESTAÑA 3: RESUMEN CLAVE Y PREGUNTAS TRAMPA
# ------------------------------------------
with tab_summary:
    st.header("Síntesis Estratégica y Trampas de Examen")
    st.markdown("Obtén los puntos esenciales del temario junto con un análisis de los errores conceptuales y preguntas trampa más frecuentes en los exámenes.")

    if st.button("✨ Analizar Temario y Generar Resumen", type="primary"):
        with st.spinner("Sintetizando conceptos clave y detectando posibles trampas de examen..."):
            prompt_summary = f"""
            Analiza el siguiente temario académico con mentalidad de examinador experto y genera:
            1. 'puntos_clave': Una lista concisa y estructurada de los conceptos, fórmulas, leyes o definiciones más importantes que el alumno debe dominar obligatoriamente.
            2. 'trampas_examen': Una lista de "trampas típicas de examen" basadas en este temario (confusiones habituales entre términos parecidos, excepciones a la regla, matices sutiles donde los alumnos suelen fallar) y cómo evitar caer en ellas.
            
            TEMARIO DE REFERENCIA:
            {st.session_state.study_text[:65000]}
            """
            try:
                response = call_gemini_with_retry(
                    client=client,
                    primary_model=MODEL_ID,
                    contents=prompt_summary,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ResumenYTrampasSchema,
                        temperature=0.3
                    )
                )
                st.session_state.summary_data = json.loads(response.text)
                st.rerun()
            except APIError as e:
                st.error(f"Alta demanda temporal en Google AI. Inténtalo de nuevo en unos segundos: {e}")
            except Exception as e:
                st.error(f"Error al generar el resumen: {e}")

    if st.session_state.summary_data:
        sum_data = st.session_state.summary_data
        col_res, col_tramp = st.columns([1, 1])

        with col_res:
            st.subheader("📌 Conceptos Clave del Temario")
            with st.container(border=True):
                for punto in sum_data.get("puntos_clave", []):
                    st.markdown(f"- {punto}")

        with col_tramp:
            st.subheader("⚠️ Trampas Típicas de Examen")
            for trampa in sum_data.get("trampas_examen", []):
                with st.container(border=True):
                    st.markdown(f"#### 🎯 {trampa.get('concepto', '')}")
                    st.markdown(f"❌ **Trampa / Error frecuente:** {trampa.get('error_comun', '')}")
                    st.markdown(f"💡 **Cómo evitarlo:** {trampa.get('como_evitarlo', '')}")

# ------------------------------------------
# PESTAÑA 4: TUTOR INTERACTIVO (CHAT)
# ------------------------------------------
with tab_tutor:
    col_chat_title, col_chat_clear = st.columns([4, 1])
    with col_chat_title:
        st.header("💬 Tutor Académico Personal")
        st.caption("Haz preguntas específicas sobre tus apuntes, pide ejemplos prácticos o solicita analogías para entender partes complejas.")
    with col_chat_clear:
        st.write("")
        if st.button("🧹 Borrar Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    chat_container = st.container(height=480)
    with chat_container:
        if not st.session_state.chat_history:
            with st.chat_message("assistant"):
                st.markdown("¡Hola! He leído tus apuntes. ¿Qué concepto te gustaría que te explique con más detalle o con un ejemplo práctico?")
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_query = st.chat_input("Escribe tu duda sobre el temario aquí...")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_query)

            system_instruction = f"""
            Eres un tutor académico pedagógico, cercano y riguroso. Tu objetivo es ayudar al estudiante a comprender el temario cargado.
            
            Reglas de actuación:
            1. Basa tus respuestas prioritariamente en el TEMARIO DE REFERENCIA adjunto.
            2. Si el alumno pregunta algo que no aparece explícitamente en los apuntes pero está directamente relacionado, puedes explicarlo indicando claramente que es información complementaria a sus apuntes.
            3. Utiliza ejemplos claros, formato Markdown estructurado y comprueba si al alumno le ha quedado claro.
            
            TEMARIO DE REFERENCIA:
            {st.session_state.study_text[:65000]}
            """

            contents_payload = []
            for m in st.session_state.chat_history:
                api_role = "user" if m["role"] == "user" else "model"
                contents_payload.append(
                    types.Content(
                        role=api_role,
                        parts=[types.Part.from_text(text=m["content"])]
                    )
                )

            with st.chat_message("assistant"):
                with st.spinner("Consultando apuntes y elaborando respuesta..."):
                    try:
                        response = call_gemini_with_retry(
                            client=client,
                            primary_model=MODEL_ID,
                            contents=contents_payload,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                temperature=0.4
                            )
                        )
                        assistant_reply = response.text
                        st.markdown(assistant_reply)
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
                    except APIError as e:
                        st.error(f"Alta demanda temporal en Google AI. Por favor, envía tu pregunta de nuevo: {e}")
                    except Exception as e:
                        st.error(f"Ocurrió un error inesperado en el chat: {e}")