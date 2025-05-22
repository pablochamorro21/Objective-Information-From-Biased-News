import streamlit as st
import time
from PIL import Image

# ----------------------------------------------
# Configuración de la aplicación
# ----------------------------------------------
st.set_page_config(
    page_title="Generador de Noticias Sin Sesgo",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Carga de assets
logo = Image.open("logo.png")
pipeline_img = Image.open("model_pipeline.png")

# Textos de ejemplo (MVP)
el_mundo_text = """
El Partido Popular ha salido este lunes a desmentir que tenga abiertas negociaciones con Vox para su incorporación al gobierno municipal del Ayuntamiento de Sevilla: ""Ni la dirección nacional del PP ni la dirección autonómica del PP de Andalucía ni el PP de Sevilla ni el gobierno del Ayuntamiento de Sevilla han mantenido ninguna reunión con VOX. No existe ningún pacto y no existe ningún texto de un acuerdo que no existe (sic)"". Los populares negaban así la existencia, no ya de un acuerdo, sino siquiera de contactos con ese objetivo. Y eso, pese a que la portavoz en Sevilla de Vox, Cristina Peláez, sí sostiene que esa negociación se está produciendo aunque, de momento, sin resultados concretos. Las alarmas han saltado a raíz de la información publicada este lunes por Diario de Sevilla, donde se daban detalles de esos contactos y se apuntaba incluso a que el acuerdo estaba cerrado, pero no se daría a conocer hasta después de las elecciones europeas, que se celebrarán el próximo 9 de junio, para evitar una eventual repercusión de la alianza en la intención de voto. Vox vuelve a meter presión, por tanto, para la firma de un acuerdo al que el PP andaluz se resiste, a pesar de que el bloqueo de la política municipal es ya más que una amenaza ante la falta de mayorías en el pleno para sacar adelante un programa propio de inversiones. El gobierno del popular José Luis Sanz, que gestiona la ciudad en minoría tras las elecciones de mayo de 2023, se vio obligado a prorrogar los presupuestos ante la falta de apoyo entre los concejales de la oposición. Desde el comienzo del mandato, Vox viene reclamando una negociación que supuestamente el alcalde les prometió cuando se conocieron los resultados electorales. La investidura de José Luis Sanz salió adelante con los votos exclusivamente de los concejales del PP (14), ya que fue la lista más votada, a sólo dos ediles de la mayoría absoluta. No había posibilidad entonces de articular una candidatura alternativa, pues se hubieran tenido que sumar para ello los votos de Vox (3 ediles) a los de la izquierda del PSOE (12) y Con Andalucía (2). Pero sí hace falta una mayoría absoluta para sacar adelante las normas y proyectos importantes para la ciudad. Por ejemplo, para aprobar las últimas ordenanzas fiscales el PP contó con el apoyo de los votos del Grupo Socialista. Por contra, el proyecto de los presupuestos de 2024 se tuvo que quedar en un cajón. Prorrogar los presupuestos es una alternativa relativamente frecuente en gobiernos en minoría, pero completar un mandato sin presupuestos propios constituye una auténtica pesadilla y lastra la capacidad del gobierno para cumplir con su programa electoral. Juanma Moreno, que gobierna con mayoría absoluta en la Junta de Andalucía, lleva a gala haber frenado el avance de la ultraderecha en la comunidad, en la medida en que, hoy por hoy, Vox no forma parte de ninguno de sus gobiernos en las ocho capitales de provincia. Pero el bloqueo de Sevilla podría estar haciendo mella a nivel local. Por otro lado, no es ningún secreto que las relaciones con Vox dividen internamente al Partido Popular, donde hay un sector que considera que el partido debe "normalizar" ante su electorado la participación de la derecha populista en los gobiernos de coalición; y quienes, como el PP-A, prefieren mantener una distancia visible con los postulados de la derecha extrema. En el PP andaluz se ha impuesto este lunes la segunda opción, negando por activa y por pasiva una supuesta intención de abrir el gobierno de Sevilla a una hipotética coalición en la que Vox tendría al menos dos delegaciones, una por cada uno de los votos que el PP necesita para superar la votaciones en el plenio. El secretario general del PP-A, Antonio Repullo, ha insistido que no hay abierta ninguna negociación en el Ayuntamiento de Sevilla para la entrada de Vox en el gobierno municipal. Repullo ha tirado de la retórica para afirmar que lo que hace el PP andaluz es "trabajar para hacer la ciudad que los sevillanos quieren y necesitan". El Ayuntamiento de Sevilla tiene un gobierno "que aporta seguridad y que trabaja diariamente por el interés general de todos los sevillanos", ha añadido. Sí ha insistido en pedir de forma genérica "responsabilidad" a las fuerzas de la oposición para que sea posible disponer de un presupuesto "que sirva a los sevillanos para progresar en su ciudad". También el presidente de la Junta ha tenido que salir al paso para asegurar que no le consta que se hayan producido contactos y que desconoce la procedencia de la información que apunta al acuerdo, según informa Efe. En cambio, la portavoz de Vox en el Ayuntamiento de Sevilla, Cristina Peláez, ha hablado de negociaciones "que van por buen camino". Y también lo ha hecho el secretario general del partido, Ignacio Garriga: "Hemos estado hablando, estamos hablando y hablaremos". Aunque ha desmentido que exista un pacto ya firmado que incluya esperar a que pasen las elecciones europeas para hacerlo público. Vox no juega, ha subrayado, "a las estrategias de la vieja política". "Cuando se suscriba un pacto, se aplicará de inmediato", ha añadido.
"""

el_pais_text = """
Desde que el candidato del PP a la alcaldía de Sevilla, José Luis Sanz, se hizo con la vara de mando en las pasadas elecciones municipales sin mayoría absoluta (ostenta 14 de los 31 concejales que configuran el pleno municipal), la posibilidad de que Vox (con tres ediles) entrara a formar parte del Ayuntamiento hispalense empezó a cobrar fuerza. Esa ha sido una exigencia perenne del partido ultra para dar su apoyo a cualquier iniciativa impulsada por el gobierno en minoría de Sanz estos meses. La prórroga de los presupuestos municipales evidenció la fragilidad de uno de los principales consistorios del país dirigido por los populares y la posibilidad de hacer un hueco en las concejalías a los de Santiago Abascal empezó a verse como algo más plausible, pese a las reticencias que el regidor siempre ha manifestado en público. Esas reservaciones se han transformado este lunes en una cascada de desmentidos que han negado categóricamente que entre las cúpulas de la formación que dirige Alberto Núñez-Feijóo y la de Abascal se hayan entablado negociaciones para que Vox entre en el consistorio hispalense. La portavoz municipal del partido de extrema derecha, Cristina Peláez sí que lo ha confirmado. "Hay conversaciones, que se les puede llamar negociaciones, que van por buen camino", ha afirmado Peláez. En Barcelona, el secretario general del partido ultra, Ignacio Garriga, ha negado que haya un pacto cerrado para entrar en el gobierno municipal de Sevilla, aunque ha confirmado la existencia de negociaciones. "Hemos hablado, estamos hablando y hablaremos con el PP para lograr lo mejor para los sevillanos. Conversaciones las hay desde el día después de las elecciones [municipales]", ha dicho. Ha dejado claro, sin embargo, que aún no se ha alcanzado ningún acuerdo porque "si lo hubiera, lo conocerían. Lo que no haremos es retrasar la publicación de un pacto porque haya elecciones", ha añadido, informa Miguel González. Mucho más categórica ha sido la negativa de los populares. "Ni la dirección nacional del PP, ni la dirección autonómica del PP de Andalucía, ni el PP de Sevilla ni el gobierno del Ayuntamiento de Sevilla han mantenido ninguna reunión con Vox. No existe ningún pacto y no existe ningún texto de un acuerdo que no existe", ha respondido el PP andaluz a través de un escueto mensaje. Una afirmación que después repetía en público su secretario general en Andalucía, Antonio Repullo, y el portavoz del partido en el Ayuntamiento sevillano, Juan Bueno. Ambas formaciones reaccionaban a la información publicada en Diario de Sevilla en la que avanza un acuerdo para la entrada de concejales de Vox en el Gobierno municipal el próximo junio o julio —tras las elecciones europeas, en todo caso―. El pacto citado, que se ha estado gestando durante varios meses, según el diario, sería similar al que la formación ultra difundió la semana pasada para entrar en el Ayuntamiento de Segovia y que también negó el PP. "Se trata de una intoxicación de Vox. Es todo una pura invención, que les conviene a ellos, si hubieran querido entrar en el Ayuntamiento de Sevilla podrían haberlo hecho hace tiempo", indica a EL PAÍS una fuente de la cúpula de los populares en Andalucía. "El Ayuntamiento de Sevilla tiene un gobierno que aporta seguridad y que trabaja diariamente por el interés general de todos los sevillanos", ha recalcado Repullo esta mañana, quien también ha negado que pueda llegarse a un eventual acuerdo con Vox después de las elecciones europeas. En la misma línea se ha manifestado, minutos después, el portavoz municipal del PP: "No sé lo que pasará dentro de dos años, como ustedes comprenderán, pero lo cierto es que a día de hoy no hay pacto". Durante la campaña electoral, Sanz no se opuso tajantemente a incluir a Vox dentro del Gobierno municipal, aunque siempre ha defendido que prefiere gobernar en minoría. Fuentes cercanas a su equipo siempre han reconocido que esa decisión no está en manos del alcalde, sino que está a expensas de lo que mejor le convenga a los intereses nacionales del partido. El PP de Moreno tampoco oculta que les conviene más no compartir la vara de mando con la formación ultra, y más a las puertas de las elecciones europeas, donde el dirigente andaluz volverá a enarbolar la bandera de la moderación para captar el voto socialista desencantado. Fuentes de la dirección de los populares andaluces sostienen que quien tiene más que perder de cara a la opinión pública es el partido de Abascal. "Serán ellos los que tendrán que explicar por qué prefieren que gobierne el PSOE a nosotros", indican sobre las eventuales consecuencias de su falta de apoyo a los populares. Sin embargo, tampoco esconden que la continua oposición a la mayoría de las iniciativas del PP en el Ayuntamiento —de hecho, buena parte de las medidas de mayor calado de Sanz han salido adelante gracias a la abstención del PSOE, como la aprobación de las ordenanzas fiscales, las últimas modificaciones presupuestarias o la subida de la tarifa del agua— ponen en evidencia su debilidad a la hora de gobernar, máxime en la capital andaluza, justo lo que le ha recordado Peláez esta mañana. Esa parálisis también la acusan en Linares (Jaén), otro de los grandes municipios donde el PP gobierna en minoría y donde la oposición de V ox también ha impedido que saquen adelante sus presupuestos. Lo sucedido la semana pasada en Segovia se distingue del caso sevillano en que, a diferencia de la capital castellana, en Sevilla ambas fuerzas coinciden en que no hay ningún acuerdo escrito, mientras que la portavoz municipal en el consistorio segoviano de Vox, Esther Núñez, sí difundió un documento de 25 puntos firmados por los dos partidos, una especie de programa en la sombra que la extrema derecha quería sacar a la luz, informa Juan Navarro. La edil convocó a los medios para informar de un “pacto de gobierno”, pero quedó aparentemente en una bravata porque nadie de sus potenciales socios lo secundó. Núñez habló claro: percibía estrategias “electoralistas” en el silencio de sus posibles socios, sintiéndose “estafados y engañados”. El equipo del alcalde de Segovia, José Mazarías, calla ante graves acusaciones de que la dirección general del PP llamó para paralizar la visibilización de ese pacto con las elecciones catalanas y europeas en el calendario inminente. La dirección nacional y la autonómica también han optado por guardar silencio, mientras que el PSOE ha pedido la dimisión de Mazarías. La portavoz de V ox en Sevilla también ha querido desvincular esas supuestas negociaciones de cualquier condicionamiento con cualquier proceso electoral.
"""

rewritten_text = """
El Partido Popular (PP) y Vox están en negociaciones para incorporar al gobierno municipal del Ayuntamiento de Sevilla, aunque la dirección nacional y autonómica del PP ha negado cualquier tipo de contacto o acuerdo con la formación ultra. La portavoz de Vox en Sevilla, Cristina Peláez, ha reconocido que las negociaciones están en marcha, pero no hay resultados concretos aún.
El gobierno del PP en Sevilla, encabezado por José Luis Sanz, ha prorrogado los presupuestos municipales debido a la falta de apoyo de la oposición. El PP necesita el apoyo de Vox para superar la mayoría absoluta en el pleno del Ayuntamiento de Sevilla.
Juanma Moreno, gobernador de la Junta de Andalucía, ha logrado frenar el avance de Vox en la comunidad andaluza, pero el bloqueo de Sevilla podría ser una amenaza para el
gobierno local.
El candidato del PP a la alcaldía de Sevilla, José Luis Sanz, obtuvo la vara de mando en las elecciones municipales sin mayoría absoluta. La posibilidad de que Vox entre en el Ayuntamiento de Sevilla ha aumentado a raíz de la prórroga de los presupuestos municipales. La portavoz municipal de Vox, Cristina Peláez, ha confirmado que se están manteniendo negociaciones con el PP para que Vox entre en el consistorio hispalense. El PP andaluz ha negado cualquier acuerdo o pacto con Vox, y ha afirmado que no se ha alcanzado ninguna reunión o conversación con la formación ultra. Fuentes cercanas al equipo de Sanz han reconocido que el alcalde pretendiente está dispuesto a negociar con Vox para lograr la mayoría absoluta en el Ayuntamiento de Sevilla. Sin embargo, el PP andaluz ha reiterado que no existe ningún acuerdo o pacto con Vox.
"""


# ----------------------------------------------
# Estilos CSS
# ----------------------------------------------
st.markdown(
    """
    <style>
      .stApp { background-color: #f5f7fa; }
      .sidebar .sidebar-content { background-color: #ffffff; padding: 1rem; }
      .big-title { font-size:2.5rem !important; color:#2c3e50 !important; margin-bottom:0.5rem; }
      .section-title { color:#34495e; margin-top:2rem; }
      .tech-code { font-size:1rem; font-weight:bold; color:#27ae60; }
    </style>
    """, unsafe_allow_html=True
)

# ----------------------------------------------
# Barra lateral de navegación & contenido opcional
# ----------------------------------------------
with st.sidebar:
    st.image(logo, width=100)
    st.title("Menú")
    page = st.radio("", ["Inicio", "Generar", "Equipo"] )

# ----------------------------------------------
# Página: Inicio
# ----------------------------------------------
if page == "Inicio":
    header_col1, header_col2 = st.columns([9, 1])
    header_col1.markdown("<h1 class='big-title'>Generador de Noticias Sin Sesgo 📰</h1>", unsafe_allow_html=True)
    header_col2.image(logo, width=100)
    header_col1.write(
    """
    ¡Bienvenidos al Generador de Noticias Sin Sesgo!
    
    En este prototipo demostrativo, puedes combinar información de **El Mundo** y **El País** para generar una narrativa objetiva, equilibrada y libre de inclinaciones políticas. A través de nuestro proceso, extraemos, fusionamos y reescribimos artículos para cuantificar y minimizar el sesgo político.
    
    Te invitamos a explorar nuestras funcionalidades, generar noticias neutras y descubrir cómo las métricas de sesgo pueden ayudarte a entender la influencia de distintos enfoques editoriales.
    
    Siéntete libre de recorrer cada sección del menú y no dudes en contactarnos si deseas más información o tienes sugerencias.
    """
)
    
    # Detalles técnicos
    st.expander("Ver detalles técnicos del pipeline").markdown(
"""
**1. Data Collection:**
- <span class='tech-code'>scraper_elmundo.py</span>: Obtiene artículos de política de El Mundo.
- <span class='tech-code'>scraper_elpais.py</span>: Obtiene artículos de política de El País.

<span style="display:block; height:1rem;"></span>
**2. Article Matching:**
- <span class='tech-code'>all_similarity.py</span>: Empareja artículos por fecha y similitud semántica.

<span style="display:block; height:1rem;"></span>
**3. Classifier Training:**
- <span class='tech-code'>classifier.py</span>: Entrena un BiLSTM para detectar sesgo político.

<span style="display:block; height:1rem;"></span>
**4. Generative RL Fine-Tuning:**
- <span class='tech-code'>gen_model_single.py</span> y <span class='tech-code'>gen_model_pairs.py</span>: Entrenan un modelo generativo usando la puntuación del clasificador como modelo de recompensa.

<span style="display:block; height:1rem;"></span>
**5. Evaluación:**
- <span class='tech-code'>gen_model_testing.py</span>: Mide la reducción de sesgo en un conjunto de noticias.
""", unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("<h2 class='section-title'>📈 Pipeline del Modelo Final</h2>", unsafe_allow_html=True)
    st.image(pipeline_img, width=600)

# ----------------------------------------------
# Página: Generar
# ----------------------------------------------
elif page == "Generar":
    header_col1, header_col2 = st.columns([9, 1])
    header_col1.markdown("<h2 class='section-title'>🔨 Ejecutar Modelo</h2>", unsafe_allow_html=True)
    header_col2.image(logo, width=100)
    header_col1.write("Genera una noticia neutral paso a paso.")
    st.markdown("---")

    st.markdown("<h2 class='section-title'>🚀 Proceso en 4 pasos</h2>", unsafe_allow_html=True)
    cols = st.columns(4)
    icons = ["🔗", "🔍", "✍️", "📊"]
    steps = [
        "Ingresar URLs de artículos",
        "Verificar fecha y tema",
        "Reescribir en estilo neutral",
        "Calcular reducción de sesgo"
    ]
    for col, icon, text in zip(cols, icons, steps):
        col.markdown(f"<h3 style='text-align:center'>{icon}</h3>", unsafe_allow_html=True)
        col.write(text)

    with st.form(key="input_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        url_mundo = col1.text_input("URL artículo El Mundo")
        url_pais = col2.text_input("URL artículo El País")
        submitted = st.form_submit_button("Extraer y Generar")

    if submitted:
        if not url_mundo or not url_pais:
            st.error("Por favor, ingresa ambas URLs.")
        else:
            st.info("🔄 Extrayendo artículos...")
            ext_progress = st.progress(0)
            for pct in range(0, 101, 25):
                time.sleep(0.3)
                ext_progress.progress(pct)
            st.success("✅ Artículos verificados")

            st.markdown("<h3>📰 Texto completo de El Mundo</h3>", unsafe_allow_html=True)
            st.write(el_mundo_text)
            st.markdown("<h3>📰 Texto completo de El País</h3>", unsafe_allow_html=True)
            st.write(el_pais_text)
            st.markdown("---")

            st.info("🔄 Generando artículo neutral...")
            gen_progress = st.progress(0)
            for pct in range(0, 101, 20):
                time.sleep(2.5)
                gen_progress.progress(pct)

            st.markdown("**Artículo Neutral Generado:**")
            st.write(rewritten_text)

            cols = st.columns(3)
            cols[0].metric("Sesgo político original", "0.8832")
            cols[1].metric("Sesgo político generado", "0.7524")
            cols[2].markdown(
                "<p style='margin:0;font-weight:bold;'>Reducción del sesgo</p>"
                "<h3 style='color:darkgreen;margin:0;'>13%</h3>",
                unsafe_allow_html=True
            )

# ----------------------------------------------
# Página: Equipo
# ----------------------------------------------
elif page == "Equipo":
    header_col1, header_col2 = st.columns([9, 1])
    header_col1.markdown("<h2 class='section-title'>🙋‍♂️🙋‍♀️ ¿Quiénes somos?</h2>", unsafe_allow_html=True)
    header_col2.image(logo, width=100)
    header_col1.write(
        "Somos un equipo interdisciplinar de IE School of Sciences and Technology, "
        "apasionados por la neutralidad informativa y el análisis de datos."
    )
    st.markdown("---")

    members = [
        {
            "name": "Pablo Chamorro Casero",
            "role": "Estudiante IE University",
            "img": "pablo_chamorro.png",
            "contact": "pchamorro.ieu2020@student.ie.edu",
            "formation": "Doble Grado en Filosofía, Política, Derecho & Economía + Análisis de Datos"
        },
        {
            "name": "Alejandro Martínez",
            "role": "Profesor Adjunto IE School of Sciences & Tech",
            "img": "alejandro_martinez.png",
            "contact": "amartinezm@faculty.ie.edu",
            "formation": "Grado en Psicología, Máster en Estadística (UAM), MBA (EOI), Doctorado en Psicología (UAM)"
        }
    ]
    cols = st.columns(len(members))
    for col, m in zip(cols, members):
        col.image(m["img"], width=150)
        col.subheader(m["name"])
        col.write(f"**{m['role']}**")
        col.write(m["formation"])
        col.write(f"✉️ {m['contact']}")
