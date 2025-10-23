import streamlit as st
import pandas as pd
from textblob import TextBlob
import re
from googletrans import Translator
import requests
import json

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Sentimientos Avanzado",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #a8e6cf 0%, #56ab2f 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #ff9a9e 0%, #f54d4d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Fondo con gradiente
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Título y descripción
st.markdown('<h1 class="main-header">🧠 Analizador de Sentimientos Avanzado</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #555;">
    Esta aplicación utiliza análisis de texto avanzado para determinar el sentimiento, subjetividad 
    y características lingüísticas de cualquier texto que ingreses.
    </p>
</div>
""", unsafe_allow_html=True)

# URLs de animaciones Lottie para diferentes sentimientos
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

# Barra lateral mejorada
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
        <h2 style="margin: 0; text-align: center;">⚙️ Opciones</h2>
    </div>
    """, unsafe_allow_html=True)
    
    modo = st.selectbox(
        "Selecciona el modo de entrada:",
        ["Texto directo", "Archivo de texto"],
        key="modo_selector"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.9); padding: 1rem; border-radius: 10px;">
        <h4 style="color: #333; margin-top: 0;">📈 Escalas de Análisis</h4>
        <p style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">
        <strong>Sentimiento:</strong> -1 (Negativo) a 1 (Positivo)
        </p>
        <p style="color: #666; font-size: 0.9rem; margin-bottom: 0;">
        <strong>Subjetividad:</strong> 0 (Objetivo) a 1 (Subjetivo)
        </p>
    </div>
    """, unsafe_allow_html=True)

# Función para contar palabras sin depender de NLTK
def contar_palabras(texto):
    # Lista básica de palabras vacías en español e inglés
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
    
    # Limpiar y tokenizar texto
    palabras = re.findall(r'\b\w+\b', texto.lower())
    
    # Filtrar palabras vacías y contar frecuencias
    palabras_filtradas = [palabra for palabra in palabras 
                         if palabra not in stop_words and len(palabra) > 2]
    
    # Contar frecuencias
    contador = {}
    for palabra in palabras_filtradas:
        contador[palabra] = contador.get(palabra, 0) + 1
    
    # Ordenar por frecuencia
    contador_ordenado = dict(sorted(contador.items(), key=lambda x: x[1], reverse=True))
    
    return contador_ordenado, palabras_filtradas

# Inicializar el traductor
translator = Translator()

# Función para traducir texto del español al inglés
def traducir_texto(texto):
    try:
        traduccion = translator.translate(texto, src='es', dest='en')
        return traduccion.text
    except Exception as e:
        st.error(f"Error al traducir: {e}")
        return texto  # Devolver el texto original si falla la traducción

# Función para procesar el texto con TextBlob (versión con traducción)
def procesar_texto(texto):
    # Guardar el texto original
    texto_original = texto
    
    # Traducir el texto al inglés para mejor análisis
    texto_ingles = traducir_texto(texto)
    
    # Analizar el texto traducido con TextBlob
    blob = TextBlob(texto_ingles)
    
    # Análisis de sentimiento (esto no requiere corpus adicionales)
    sentimiento = blob.sentiment.polarity
    subjetividad = blob.sentiment.subjectivity
    
    # Extraer frases de manera simplificada (del texto original)
    frases_originales = [frase.strip() for frase in re.split(r'[.!?]+', texto_original) if frase.strip()]
    
    # Extraer frases del texto traducido
    frases_traducidas = [frase.strip() for frase in re.split(r'[.!?]+', texto_ingles) if frase.strip()]
    
    # Combinar frases originales y traducidas
    frases_combinadas = []
    for i in range(min(len(frases_originales), len(frases_traducidas))):
        frases_combinadas.append({
            "original": frases_originales[i],
            "traducido": frases_traducidas[i]
        })
    
    # Contar palabras con nuestra función simplificada (en el texto traducido)
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

# Función para mostrar animación Lottie según el sentimiento
def mostrar_animacion_sentimiento(sentimiento):
    if sentimiento > 0.05:
        lottie_json = load_lottie_url(LOTTIE_URLS["positive"])
        sentimiento_clase = "sentiment-positive"
        etiqueta = "POSITIVO"
        emoji = "😊"
    elif sentimiento < -0.05:
        lottie_json = load_lottie_url(LOTTIE_URLS["negative"])
        sentimiento_clase = "sentiment-negative"
        etiqueta = "NEGATIVO"
        emoji = "😟"
    else:
        lottie_json = load_lottie_url(LOTTIE_URLS["neutral"])
        sentimiento_clase = "sentiment-neutral"
        etiqueta = "NEUTRAL"
        emoji = "😐"
    
    # Mostrar etiqueta de sentimiento
    st.markdown(f"""
    <div class="{sentimiento_clase}">
        <h2 style="margin: 0; font-size: 1.8rem;">{emoji} Sentimiento: {etiqueta}</h2>
        <p style="margin: 0; font-size: 1.2rem;">Valor: {sentimiento:.3f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar animación Lottie si está disponible
    if lottie_json:
        # En Streamlit, podemos usar componentes para Lottie, pero aquí usamos un placeholder
        # En una implementación real, usaríamos: from streamlit_lottie import st_lottie
        st.markdown(f"""
        <div style="text-align: center; margin: 1rem 0;">
            <p>🎭 Animación de {etiqueta.lower()} cargada</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align: center; margin: 1rem 0;">
            <p>⚠️ No se pudo cargar la animación</p>
        </div>
        """, unsafe_allow_html=True)

# Función para crear visualizaciones usando componentes nativos de Streamlit
def crear_visualizaciones(resultados):
    # Mostrar animación de sentimiento
    mostrar_animacion_sentimiento(resultados["sentimiento"])
    
    # Métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin-top: 0; color: #333;">📊 Sentimiento</h3>
            <p style="font-size: 1.5rem; font-weight: bold; color: #667eea; margin-bottom: 0;">
            {:.3f}
            </p>
        </div>
        """.format(resultados["sentimiento"]), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin-top: 0; color: #333;">🎯 Subjetividad</h3>
            <p style="font-size: 1.5rem; font-weight: bold; color: #667eea; margin-bottom: 0;">
            {:.3f}
            </p>
        </div>
        """.format(resultados["subjetividad"]), unsafe_allow_html=True)
    
    with col3:
        num_palabras = len(resultados["palabras"])
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin-top: 0; color: #333;">📝 Palabras Útiles</h3>
            <p style="font-size: 1.5rem; font-weight: bold; color: #667eea; margin-bottom: 0;">
            {}
            </p>
        </div>
        """.format(num_palabras), unsafe_allow_html=True)
    
    # Visualización de sentimiento y subjetividad con barras de progreso
    st.markdown('<div class="sub-header">📈 Análisis Detallado</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="margin-top: 0; color: #333;">Sentimiento</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Normalizar valores para mostrarlos en barras de progreso
        # Sentimiento va de -1 a 1, lo normalizamos a 0-1 para la barra
        sentimiento_norm = (resultados["sentimiento"] + 1) / 2
        
        st.progress(sentimiento_norm)
        
        if resultados["sentimiento"] > 0.05:
            st.success(f"📈 Predominantemente Positivo ({resultados['sentimiento']:.3f})")
        elif resultados["sentimiento"] < -0.05:
            st.error(f"📉 Predominantemente Negativo ({resultados['sentimiento']:.3f})")
        else:
            st.info(f"📊 Mayormente Neutral ({resultados['sentimiento']:.3f})")
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="margin-top: 0; color: #333;">Subjetividad</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Subjetividad ya está en el rango 0-1
        st.progress(resultados["subjetividad"])
        
        if resultados["subjetividad"] > 0.5:
            st.warning(f"💭 Texto Subjetivo ({resultados['subjetividad']:.3f})")
        else:
            st.info(f"📋 Texto Objetivo ({resultados['subjetividad']:.3f})")
    
    # Palabras más frecuentes usando chart de Streamlit
    st.markdown('<div class="sub-header">🔤 Palabras Más Frecuentes</div>', unsafe_allow_html=True)
    
    if resultados["contador_palabras"]:
        palabras_top = dict(list(resultados["contador_palabras"].items())[:10])
        
        # Crear un dataframe para mostrar las palabras más frecuentes
        df_palabras = pd.DataFrame({
            'Palabra': list(palabras_top.keys()),
            'Frecuencia': list(palabras_top.values())
        })
        
        # Mostrar tabla y gráfico
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(df_palabras, use_container_width=True)
        
        with col2:
            st.bar_chart(df_palabras.set_index('Palabra'))
    else:
        st.info("No se encontraron palabras significativas para analizar.")
    
    # Mostrar texto traducido
    st.markdown('<div class="sub-header">🌐 Traducción</div>', unsafe_allow_html=True)
    
    with st.expander("Ver traducción completa", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Texto Original (Español):**")
            st.text_area("", resultados["texto_original"], height=150, key="original_text")
        with col2:
            st.markdown("**Texto Traducido (Inglés):**")
            st.text_area("", resultados["texto_traducido"], height=150, key="translated_text")
    
    # Análisis de frases
    st.markdown('<div class="sub-header">📖 Análisis de Frases</div>', unsafe_allow_html=True)
    
    if resultados["frases"]:
        for i, frase_dict in enumerate(resultados["frases"][:10], 1):
            frase_original = frase_dict["original"]
            frase_traducida = frase_dict["traducido"]
            
            try:
                blob_frase = TextBlob(frase_traducida)
                sentimiento = blob_frase.sentiment.polarity
                
                if sentimiento > 0.05:
                    emoji = "😊"
                    color = "#a8e6cf"
                elif sentimiento < -0.05:
                    emoji = "😟"
                    color = "#ff9a9e"
                else:
                    emoji = "😐"
                    color = "#a1c4fd"
                
                st.markdown(f"""
                <div style="background: {color}; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                    <p style="margin: 0; font-weight: bold;">{i}. {emoji} Sentimiento: {sentimiento:.3f}</p>
                    <p style="margin: 0.5rem 0 0 0;"><strong>Original:</strong> "{frase_original}"</p>
                    <p style="margin: 0;"><strong>Traducción:</strong> "{frase_traducida}"</p>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown(f"""
                <div style="background: #f0f0f0; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                    <p style="margin: 0; font-weight: bold;">{i}. Análisis no disponible</p>
                    <p style="margin: 0.5rem 0 0 0;"><strong>Original:</strong> "{frase_original}"</p>
                    <p style="margin: 0;"><strong>Traducción:</strong> "{frase_traducida}"</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No se detectaron frases para analizar.")

# Lógica principal según el modo seleccionado
if modo == "Texto directo":
    st.markdown('<div class="sub-header">✍️ Ingresa tu Texto para Analizar</div>', unsafe_allow_html=True)
    
    texto = st.text_area(
        "", 
        height=200, 
        placeholder="Escribe o pega aquí el texto que deseas analizar...",
        key="text_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🔍 Analizar Texto", use_container_width=True):
            if texto.strip():
                with st.spinner("Analizando texto..."):
                    resultados = procesar_texto(texto)
                    crear_visualizaciones(resultados)
            else:
                st.warning("Por favor, ingresa algún texto para analizar.")

elif modo == "Archivo de texto":
    st.markdown('<div class="sub-header">📁 Carga un Archivo de Texto</div>', unsafe_allow_html=True)
    
    archivo = st.file_uploader("", type=["txt", "csv", "md"], key="file_uploader")
    
    if archivo is not None:
        try:
            contenido = archivo.getvalue().decode("utf-8")
            with st.expander("Vista previa del contenido", expanded=False):
                st.text(contenido[:1000] + ("..." if len(contenido) > 1000 else ""))
            
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("🔍 Analizar Archivo", use_container_width=True):
                    with st.spinner("Analizando archivo..."):
                        resultados = procesar_texto(contenido)
                        crear_visualizaciones(resultados)
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

# Información adicional
with st.expander("📚 Información sobre el Análisis", expanded=False):
    st.markdown("""
# Pie de página
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p style="margin: 0;">Desarrollado con ❤️ usando Streamlit y TextBlob | 
    <a href="#" style="color: #667eea; text-decoration: none;">Analizador de Sentimientos Avanzado</a></p>
</div>
""", unsafe_allow_html=True)
