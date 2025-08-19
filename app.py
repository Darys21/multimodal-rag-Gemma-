import os
import sys
import tempfile
import base64
from pathlib import Path

import streamlit as st
from PIL import Image
from pyvis.network import Network
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import NLP_Project_Graph_Based_RAG  # noqa: E402  (import après modification du path)

st.set_page_config(
    page_title="RAG Multimodal avec Graphe de Connaissances",
    page_icon="🔍",
    layout="wide"
)

# ------------------------------------------------------------------
# 3) Style CSS optionnel
# ------------------------------------------------------------------
st.markdown("""
<style>
    .main {max-width: 1200px; padding: 2rem;}
    .title {color: #1E88E5; text-align: center; margin-bottom: 2rem;}
    .result-section {margin-top: 2rem; padding: 1.5rem; border-radius: 0.5rem; background-color: #f8f9fa;}
</style>
""", unsafe_allow_html=True)

# 4) Titre
st.title("🔍 RAG Multimodal avec Graphe de Connaissances")
st.markdown("---")

# 5) Sidebar
with st.sidebar:
    st.header("Configuration")

    query_mode = st.radio("Type de requête :", ["Texte", "Image"], index=0)

    if query_mode == "Texte":
        query_text = st.text_area(
            "Entrez votre requête textuelle :",
            placeholder="Ex : Où travaille Alice ?",
            height=100
        )
        uploaded_file = None
    else:
        query_text = None
        uploaded_file = st.file_uploader(
            "Téléchargez une image :",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Aperçu", use_column_width=True)

    st.markdown("---")
    st.header("Paramètres")

    max_depth = st.slider(
        "Profondeur max du graphe",
        min_value=1,
        max_value=3,
        value=2
    )
    top_k = st.slider(
        "Nombre de résultats (top-k)",
        min_value=3,
        max_value=10,
        value=6
    )

    submit_button = st.button("Lancer la requête", type="primary")

if submit_button:
    with st.spinner("Traitement en cours…"):
        # ----------------------------------------------------------
        # A) Requête TEXT
        # ----------------------------------------------------------
        if query_mode == "Texte" and query_text:
            answer, subgraph = backend.search_from_txt_with_rag_context(
                query_text,
                max_graph_depth=max_depth,
                num_top_results=top_k
            )
        # ----------------------------------------------------------
        # B) Requête IMAGE
        # ----------------------------------------------------------
        elif query_mode == "Image" and uploaded_file is not None:
            # On enregistre l’image dans le dossier attendu par le backend
            save_dir = Path("user_image_search")
            save_dir.mkdir(exist_ok=True)
            img_path = save_dir / uploaded_file.name
            Image.open(uploaded_file).save(img_path)
            answer, subgraph = backend.search_from_img_with_rag_context(
                str(img_path),
                max_graph_depth=max_depth,
                num_top_results=top_k
            )
        else:
            st.warning("Veuillez saisir une question ou sélectionner une image.")
            st.stop()

    st.success("Requête traitée avec succès !")

    # ---- Réponse textuelle ------------------------------------------
    with st.expander("📝 Réponse générée", expanded=True):
        st.write(answer)

    # ---- Visualisation graphe ---------------------------------------
    st.markdown("### 🕸️ Contexte du graphe (sous-graphe)")
    if not subgraph:
        st.info("Aucun sous-graphe trouvé.")
    else:
        # Construction d’un graphe PyVis à partir du résultat Neo4j
        net = Network(
            height="600px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=True
        )

        nodes, edges = set(), set()

        for record in subgraph:
            # record["a"] est le nœud source
            # record["r"] est une *liste* de relations
            for rel in record.get("r", []):
                start = rel.start_node
                end = rel.end_node

                # Ajout des nœuds (id, label, properties)
                nodes.add((
                    start.element_id,
                    list(start.labels)[0] if start.labels else "Node",
                    dict(start)
                ))
                nodes.add((
                    end.element_id,
                    list(end.labels)[0] if end.labels else "Node",
                    dict(end)
                ))

                # Ajout de l’arête
                edges.add((
                    start.element_id,
                    end.element_id,
                    rel.type
                ))

        # Ajout dans PyVis
        for (n_id, label, props) in nodes:
            net.add_node(n_id, label=label, title=str(props), size=25)

        for (src, dst, typ) in edges:
            net.add_edge(src, dst, label=typ)

        # Export HTML temporaire + affichage
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, "r", encoding="utf-8") as f:
                components.html(f.read(), height=620)
            os.unlink(tmp.name)

with st.expander("❓ Comment utiliser cette application"):
    st.markdown("""
    1. Sélectionnez **Texte** ou **Image** dans la barre latérale  
    2. Rédigez votre question ou téléversez une image  
    3. Ajustez profondeur et top-k si besoin  
    4. Lancez la requête et explorez la réponse & le graphe !
    """)
# footer 
st.markdown("---")
st.markdown("© 2025 – RAG Multimodal avec Graphe de Connaissances")