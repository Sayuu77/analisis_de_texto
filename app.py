import streamlit as st
import pandas as pd
from textblob import TextBlob
import re
from googletrans import Translator

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Sentimientos",
    page_icon="🧠",
    layout="wide"
)

# Tema oscuro con colores emocionales
st.markdown("""
<style>
    .stApp {
        background: #0f0f23;
        color: #e0e0e0;
    }
    .main-title {
        font-size: 2.8rem;
        text-align: center;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    .subtitle {
        text-align: center;
        color: #a0a0a0;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .sentiment-positive {
        color: #4ecdc4;
        font-weight: 600;
    }
    .sentiment-negative {
        color: #ff6b6b;
        font-weight: 600;
    }
    .sentiment-neutral {
        color: #45b7d1;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .progress-container {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .section-title {
        font-size: 1.4rem;
        margin: 2rem 0 1rem 0;
        color: #ffffff;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    .phrase-analysis {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    .sidebar .sidebar-content {
        background: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-title">🧠 Analizador de Sentimientos</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Descubre las emociones detrás de tus textos</div>', unsafe_allow_html=True)

# Barra lateral minimalista
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    modo = st.selectbox(
        "Modo de entrada:",
        ["Texto directo", "Archivo de texto"]
    )
    
    st.markdown("---")
    st.markdown("Escalas")
    st.markdown("**Sentimiento:** -1 (Negativo) a 1 (Positivo)")
    st.markdown("**Subjetividad:** 0 (Objetivo) a 1 (Subjetivo)")

# Funciones de análisis (las mismas que antes)
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

translator = Translator()

def traducir_texto(texto):
    try:
        traduccion = translator.translate(texto, src='es', dest='en')
        return traduccion.text
    except:
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

def mostrar_sentimiento(sentimiento):
    if sentimiento > 0.1:
        clase = "sentiment-positive"
        emoji = "😊"
        etiqueta = "POSITIVO"
    elif sentimiento < -0.1:
        clase = "sentiment-negative"
        emoji = "😔"
        etiqueta = "NEGATIVO"
    else:
        clase = "sentiment-neutral"
        emoji = "😐"
        etiqueta = "NEUTRAL"
    
    st.markdown(f'<div class="{clase}" style="text-align: center; font-size: 2rem; margin: 2rem 0;">{emoji} {etiqueta}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value" style="text-align: center;">{sentimiento:.3f}</div>', unsafe_allow_html=True)

# Entrada principal
if modo == "Texto directo":
    texto = st.text_area(
        "Ingresa tu texto:",
        height=150,
        placeholder="Escribe o pega aquí el texto que deseas analizar..."
    )
    
    if st.button("Analizar Sentimiento"):
        if texto.strip():
            with st.spinner("Analizando emociones..."):
                resultados = procesar_texto(texto)
                
                # Resultado principal del sentimiento
                mostrar_sentimiento(resultados["sentimiento"])
                
                # Métricas rápidas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Sentimiento**")
                    st.markdown(f'<div class="metric-value">{resultados["sentimiento"]:.3f}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Subjetividad**")
                    st.markdown(f'<div class="metric-value">{resultados["subjetividad"]:.3f}</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown("**Palabras clave**")
                    st.markdown(f'<div class="metric-value">{len(resultados["palabras"])}</div>', unsafe_allow_html=True)
                
                # Barras de progreso
                st.markdown('<div class="section-title">📊 Análisis Detallado</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Intensidad del Sentimiento**")
                    sentimiento_norm = (resultados["sentimiento"] + 1) / 2
                    st.progress(sentimiento_norm)
                    st.text(f"{resultados['sentimiento']:.3f}")
                
                with col2:
                    st.markdown("**Nivel de Subjetividad**")
                    st.progress(resultados["subjetividad"])
                    st.text(f"{resultados['subjetividad']:.3f}")
                
                # Palabras frecuentes
                if resultados["contador_palabras"]:
                    st.markdown('<div class="section-title">🔤 Palabras Más Frecuentes</div>', unsafe_allow_html=True)
                    palabras_top = dict(list(resultados["contador_palabras"].items())[:8])
                    st.bar_chart(palabras_top)
                
                # Análisis de frases
                if resultados["frases"]:
                    st.markdown('<div class="section-title">💬 Análisis por Frases</div>', unsafe_allow_html=True)
                    for i, frase_dict in enumerate(resultados["frases"][:6], 1):
                        frase_traducida = frase_dict["traducido"]
                        try:
                            blob_frase = TextBlob(frase_traducida)
                            sent_frase = blob_frase.sentiment.polarity
                            
                            if sent_frase > 0.1:
                                emoji_frase = "😊"
                                color = "#4ecdc4"
                            elif sent_frase < -0.1:
                                emoji_frase = "😔"
                                color = "#ff6b6b"
                            else:
                                emoji_frase = "😐"
                                color = "#45b7d1"
                            
                            st.markdown(f'''
                            <div class="phrase-analysis">
                                <div style="color: {color}; font-weight: 600;">
                                    {emoji_frase} Frase {i} • Sentimiento: {sent_frase:.3f}
                                </div>
                                <div style="color: #b0b0b0; margin-top: 0.3rem;">
                                    "{frase_dict['original'][:100]}{'...' if len(frase_dict['original']) > 100 else ''}"
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                        except:
                            pass
        else:
            st.warning("Por favor ingresa algún texto para analizar.")

elif modo == "Archivo de texto":
    archivo = st.file_uploader("Sube un archivo de texto", type=["txt", "csv", "md"])
    
    if archivo is not None:
        try:
            contenido = archivo.getvalue().decode("utf-8")
            
            if st.button("Analizar Archivo"):
                with st.spinner("Analizando archivo..."):
                    resultados = procesar_texto(contenido)
                    
                    mostrar_sentimiento(resultados["sentimiento"])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sentimiento", f"{resultados['sentimiento']:.3f}")
                    with col2:
                        st.metric("Subjetividad", f"{resultados['subjetividad']:.3f}")
                    
                    if resultados["contador_palabras"]:
                        st.markdown("**Palabras clave más frecuentes:**")
                        palabras_top = dict(list(resultados["contador_palabras"].items())[:5])
                        for palabra, freq in palabras_top.items():
                            st.text(f"• {palabra} ({freq})")
                            
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

# Información minimalista
with st.expander("ℹ️ Acerca del análisis"):
    st.markdown("""
    **Sentimiento:** Mide la carga emocional del texto
    - **Positivo** (> 0.1): Contenido optimista y favorable
    - **Negativo** (< -0.1): Contenido pesimista o desfavorable  
    - **Neutral** (-0.1 a 0.1): Contenido equilibrado
    
    **Subjetividad:** Indica el nivel de opinión personal
    - **Alta** (> 0.5): Opiniones y juicios personales
    - **Baja** (< 0.5): Hechos e información objetiva
    """)
