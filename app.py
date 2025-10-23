import streamlit as st
import pandas as pd
from textblob import TextBlob
import re
from googletrans import Translator
import requests
import json

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Sentimientos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color principal único
COLOR_PRINCIPAL = "#2563eb"  # Azul moderno
COLOR_SECUNDARIO = "#1e40af"  # Azul más oscuro
COLOR_FONDO = "#f8fafc"      # Gris muy claro
COLOR_TEXTO = "#1e293b"      # Gris oscuro para texto

# Aplicar estilos CSS personalizados
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        color: {COLOR_PRINCIPAL};
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }}
    .sub-header {{
        font-size: 1.3rem;
        color: {COLOR_SECUNDARIO};
        margin-bottom: 1rem;
        font-weight: 600;
    }}
    .stButton button {{
        background-color: {COLOR_PRINCIPAL};
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
    }}
    .stButton button:hover {{
        background-color: {COLOR_SECUNDARIO};
        transform: translateY(-1px);
    }}
    .metric-card {{
        background: white;
        border-radius: 8px;
        padding: 1.2rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid {COLOR_PRINCIPAL};
        margin-bottom: 1rem;
    }}
    .sentiment-container {{
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        text-align: center;
    }}
    .phrase-card {{
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 0.8rem;
        border-left: 3px solid {COLOR_PRINCIPAL};
    }}
    .sidebar-header {{
        background-color: {COLOR_PRINCIPAL};
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        text-align: center;
    }}
    .stApp {{
        background-color: {COLOR_FONDO};
    }}
    .text-input {{
        background: white;
        border-radius: 8px;
        padding: 1rem;
    }}
    .info-box {{
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}
</style>
""", unsafe_allow_html=True)

# Título y descripción
st.markdown('<h1 class="main-header">📊 Analizador de Sentimientos</h1>', unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; margin-bottom: 2rem; color: {COLOR_TEXTO};">
    <p style="font-size: 1.1rem;">
    Analiza el sentimiento, subjetividad y características lingüísticas de cualquier texto
    </p>
</div>
""", unsafe_allow_html=True)

# URLs de animaciones Lottie
LOTTIE_URLS = {
    "positive": "https://assets1.lottiefiles.com/packages/lf20_ukgjv2gq.json",
    "negative": "https://assets1.lottiefiles.com/packages/lf20_7sk0n0gg.json", 
    "neutral": "https://assets1.lottiefiles.com/packages/lf20_qpsnmykx.json"
}

# Función para cargar animación Lottie
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Barra lateral
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2 style="margin: 0; font-size: 1.3rem;">⚙️ Configuración</h2>
    </div>
    """, unsafe_allow_html=True)
    
    modo = st.selectbox(
        "Modo de entrada:",
        ["Texto directo", "Archivo de texto"],
        key="modo_selector"
    )
    
    st.markdown("---")
    
    st.markdown(f"""
    <div class="info-box">
        <h4 style="color: {COLOR_TEXTO}; margin-top: 0;">📈 Escalas de Análisis</h4>
        <p style="color: {COLOR_TEXTO}; font-size: 0.9rem; margin-bottom: 0.5rem;">
        <strong>Sentimiento:</strong> -1 (Negativo) a 1 (Positivo)
        </p>
        <p style="color: {COLOR_TEXTO}; font-size: 0.9rem; margin-bottom: 0;">
        <strong>Subjetividad:</strong> 0 (Objetivo) a 1 (Subjetivo)
        </p>
    </div>
    """, unsafe_allow_html=True)

# Función para contar palabras sin depender de NLTK
def contar_palabras(texto):
    stop_words = set([
        "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con", "contra",
        "cual", "cuando", "de", "del", "desde", "donde", "durante", "e", "el", "ella",
        "ellas", "ellos", "en", "entre", "era", "eras", "es", "esa", "esas", "ese",
        "eso", "esos", "esta", "estas", "este", "esto", "estos", "ha", "había", "han",
        "has", "hasta", "he", "la", "las", "le", "les", "lo", "los", "me", "mi", "mía",
        "mías", "mío", "míos", "mis", "mucho", "muchos", "muy", "nada", "ni", "no", "nos",
        "nosotras", "nosotros", "nuestra", "nuestras", "nuestro", "nuestros", "o", "os", 
        "otra", "otras", "otro", "otros", "para", "pero", "poco", "por", "porque", "que", 
        "quien", "quienes", "qué", "se", "sea", "sean", "según", "si", "sido", "sin", 
        "sobre", "sois", "somos", "son", "soy", "su", "sus", "suya", "suyas", "suyo", 
        "suyos", "también", "tanto", "te", "tenéis", "tenemos", "tener", "tengo", "ti", 
        "tiene", "tienen", "todo", "todos", "tu", "tus", "tuya", "tuyas", "tuyo", "tuyos", 
        "tú", "un", "una", "uno", "unos", "vosotras", "vosotros", "vuestra", "vuestras", 
        "vuestro", "vuestros", "y", "ya", "yo",
        # Inglés
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", 
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", 
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", 
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", 
        "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", 
        "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
        "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", 
        "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", 
        "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", 
        "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", 
        "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", 
        "the", "their", "theirs", "them", "themselves", "then", "there", "there's", 
        "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", 
        "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
        "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", 
        "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", 
        "why's", "with", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
        "your", "yours", "yourself", "yourselves"
    ])
    
    palabras = re.findall(r'\b\w+\b', texto.lower())
    palabras_filtradas = [palabra for palabra in palabras 
                         if palabra not in stop_words and len(palabra) > 2]
    
    contador = {}
    for palabra in palabras_filtradas:
        contador[palabra] = contador.get(palabra, 0) + 1
    
    contador_ordenado = dict(sorted(contador.items(), key=lambda x: x[1], reverse=True))
    
    return contador_ordenado, palabras_filtradas

# Inicializar el traductor
translator = Translator()

def traducir_texto(texto):
    try:
        traduccion = translator.translate(texto, src='es', dest='en')
        return traduccion.text
    except Exception as e:
        st.error(f"Error al traducir: {e}")
        return texto

def procesar_texto(texto):
    texto_original = texto
    texto_ingles = traducir_texto(texto)
    
    blob = TextBlob(texto_ingles)
    sentimiento = blob.sentiment.polarity
    subjetividad = blob.sentiment.subjectivity
    
    frases_originales = [frase.strip() for frase in re.split(r'[.!?]+', texto_original) if frase.strip()]
    frases_traducidas = [frase.strip() for frase in re.split(r'[.!?]+', texto_ingles) if frase.strip()]
    
    frases_combinadas = []
    for i in range(min(len(frases_originales), len(frases_traducidas))):
        frases_combinadas.append({
            "original": frases_originales[i],
            "traducido": frases_traducidas[i]
        })
    
    contador_palabras, palabras = contar_palabras(texto_ingles)
    
    return {
        "sentimiento": sentimiento,
        "subjetividad": subjetividad,
        "frases": frases_combinadas,
        "contador_palabras": contador_palabras,
        "palabras": palabras,
        "texto_original": texto_original,
        "texto_traducido": texto_ingles
    }

def mostrar_resultado_sentimiento(sentimiento):
    if sentimiento > 0.05:
        etiqueta = "POSITIVO"
        emoji = "😊"
        color = COLOR_PRINCIPAL
    elif sentimiento < -0.05:
        etiqueta = "NEGATIVO"
        emoji = "😟"
        color = COLOR_PRINCIPAL
    else:
        etiqueta = "NEUTRAL"
        emoji = "😐"
        color = COLOR_PRINCIPAL
    
    st.markdown(f"""
    <div class="sentiment-container">
        <h2 style="color: {color}; margin: 0; font-size: 1.8rem;">{emoji} {etiqueta}</h2>
        <p style="color: {COLOR_TEXTO}; font-size: 1.1rem; margin: 0.5rem 0 0 0;">
        Valor: {sentimiento:.3f}
        </p>
    </div>
    """, unsafe_allow_html=True)

def crear_visualizaciones(resultados):
    # Resultado del sentimiento
    mostrar_resultado_sentimiento(resultados["sentimiento"])
    
    # Métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin-top: 0; color: {COLOR_TEXTO}; font-size: 1rem;">Sentimiento</h3>
            <p style="font-size: 1.4rem; font-weight: bold; color: {COLOR_PRINCIPAL}; margin-bottom: 0;">
            {resultados['sentimiento']:.3f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin-top: 0; color: {COLOR_TEXTO}; font-size: 1rem;">Subjetividad</h3>
            <p style="font-size: 1.4rem; font-weight: bold; color: {COLOR_PRINCIPAL}; margin-bottom: 0;">
            {resultados['subjetividad']:.3f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        num_palabras = len(resultados["palabras"])
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin-top: 0; color: {COLOR_TEXTO}; font-size: 1rem;">Palabras Útiles</h3>
            <p style="font-size: 1.4rem; font-weight: bold; color: {COLOR_PRINCIPAL}; margin-bottom: 0;">
            {num_palabras}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Barras de progreso
    st.markdown(f'<div class="sub-header">Análisis Detallado</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 1.2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="margin-top: 0; color: {COLOR_TEXTO};">Sentimiento</h4>
        </div>
        """, unsafe_allow_html=True)
        
        sentimiento_norm = (resultados["sentimiento"] + 1) / 2
        st.progress(sentimiento_norm)
        
    with col2:
        st.markdown(f"""
        <div style="background: white; padding: 1.2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="margin-top: 0; color: {COLOR_TEXTO};">Subjetividad</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(resultados["subjetividad"])
    
    # Palabras frecuentes
    st.markdown(f'<div class="sub-header">Palabras Más Frecuentes</div>', unsafe_allow_html=True)
    
    if resultados["contador_palabras"]:
        palabras_top = dict(list(resultados["contador_palabras"].items())[:10])
        df_palabras = pd.DataFrame({
            'Palabra': list(palabras_top.keys()),
            'Frecuencia': list(palabras_top.values())
        })
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(df_palabras, use_container_width=True)
        
        with col2:
            st.bar_chart(df_palabras.set_index('Palabra'))
    
    # Traducción
    st.markdown(f'<div class="sub-header">Traducción</div>', unsafe_allow_html=True)
    
    with st.expander("Ver traducción completa", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Texto Original (Español):**")
            st.text_area("", resultados["texto_original"], height=120, key="original_text")
        with col2:
            st.markdown("**Texto Traducido (Inglés):**")
            st.text_area("", resultados["texto_traducido"], height=120, key="translated_text")
    
    # Análisis de frases
    st.markdown(f'<div class="sub-header">Análisis de Frases</div>', unsafe_allow_html=True)
    
    if resultados["frases"]:
        for i, frase_dict in enumerate(resultados["frases"][:8], 1):
            frase_original = frase_dict["original"]
            frase_traducida = frase_dict["traducido"]
            
            try:
                blob_frase = TextBlob(frase_traducida)
                sentimiento = blob_frase.sentiment.polarity
                
                if sentimiento > 0.05:
                    emoji = "😊"
                elif sentimiento < -0.05:
                    emoji = "😟"
                else:
                    emoji = "😐"
                
                st.markdown(f"""
                <div class="phrase-card">
                    <p style="margin: 0; font-weight: bold; color: {COLOR_PRINCIPAL};">
                    {i}. {emoji} Sentimiento: {sentimiento:.3f}
                    </p>
                    <p style="margin: 0.3rem 0 0 0; color: {COLOR_TEXTO};">
                    <strong>Original:</strong> {frase_original}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown(f"""
                <div class="phrase-card">
                    <p style="margin: 0; font-weight: bold; color: {COLOR_PRINCIPAL};">
                    {i}. Análisis no disponible
                    </p>
                    <p style="margin: 0.3rem 0 0 0; color: {COLOR_TEXTO};">
                    <strong>Original:</strong> {frase_original}
                    </p>
                </div>
                """, unsafe_allow_html=True)

# Lógica principal
if modo == "Texto directo":
    st.markdown(f'<div class="sub-header">Ingresa tu Texto</div>', unsafe_allow_html=True)
    
    texto = st.text_area(
        "", 
        height=150, 
        placeholder="Escribe o pega aquí el texto que deseas analizar...",
        key="text_input"
    )
    
    if st.button("Analizar Texto", key="analyze_btn"):
        if texto.strip():
            with st.spinner("Analizando texto..."):
                resultados = procesar_texto(texto)
                crear_visualizaciones(resultados)
        else:
            st.warning("Por favor, ingresa algún texto para analizar.")

elif modo == "Archivo de texto":
    st.markdown(f'<div class="sub-header">Carga un Archivo</div>', unsafe_allow_html=True)
    
    archivo = st.file_uploader("", type=["txt", "csv", "md"], key="file_uploader")
    
    if archivo is not None:
        try:
            contenido = archivo.getvalue().decode("utf-8")
            with st.expander("Vista previa del contenido", expanded=False):
                st.text(contenido[:800] + ("..." if len(contenido) > 800 else ""))
            
            if st.button("Analizar Archivo", key="analyze_file_btn"):
                with st.spinner("Analizando archivo..."):
                    resultados = procesar_texto(contenido)
                    crear_visualizaciones(resultados)
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

# Información adicional
with st.expander("Información sobre el Análisis", expanded=False):
    st.markdown(f"""
    <div class="info-box">
        <h3 style="color: {COLOR_TEXTO}; margin-top: 0;">Sobre el análisis de texto</h3>
    """, unsafe_allow_html=True)
        <h4 style="color: {COLOR_TEXTO};">Escalas de Medición</h4>
        <ul>
            <li><strong>Sentimiento</strong>: Varía de -1 (muy negativo) a 1 (muy positivo)</li>
            <li><strong>Subjetividad</strong>: Varía de 0 (muy objetivo) a 1 (muy subjetivo)</li>
        </ul>
        
        <h4 style="color: {COLOR_TEXTO};">Tecnologías Utilizadas</h4>
        <p>Streamlit • TextBlob • Google Translate • Pandas</p>
    </div>
  

# Pie de página
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {COLOR_TEXTO}; padding: 1rem; font-size: 0.9rem;">
    <p style="margin: 0;">Desarrollado con Streamlit y TextBlob</p>
</div>
""", unsafe_allow_html=True)
