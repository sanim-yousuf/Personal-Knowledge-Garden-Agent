import streamlit as st
import os
from core import get_index, llm, ingest_and_index, generate_reflection

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Knowledge Garden",
    page_icon="🍃",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
    <style>
        body {
            background-color: #cfe2c3;
        }
        .main-title {
            font-size: 42px;
            font-weight: bold;
            color: #1b5e20;
        }
        .sub-text {
            font-size: 18px;
            color: #444;
        }
        .card {
            padding: 20px;
            border-radius: 15px;
            background-color: white;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# ================= LOAD INDEX =================
@st.cache_resource
def load_index():
    return get_index()

index = load_index()

# ================= HEADER =================
st.markdown('<p class="main-title">☘️ Knowledge Garden</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Grow your ideas, notes, and reflections 🌱</p>', unsafe_allow_html=True)

st.divider()

# ================= SIDEBAR =================
st.sidebar.title("☘️ Menu")
option = st.sidebar.radio(
    "Navigate",
    ["Home", "Add Note", "View Notes", "Ask AI", "Reflection"]
)

# ================= HOME =================
if option == "Home":
    st.markdown("### 🌿 Welcome to your Knowledge Garden")
    st.write("Store your ideas, explore them, and let AI help you grow knowledge!")

# ================= ADD NOTE =================
elif option == "Add Note":
    st.markdown("### 🍃 Add a New Note")

    uploaded_file = st.file_uploader("Upload a file")

    if uploaded_file:
        if st.button("Save to Garden"):
            ingest_and_index(uploaded_file.getbuffer(), uploaded_file.name, index)
            st.success("File added to your garden 🌱")

# ================= VIEW NOTES =================
elif option == "View Notes":
    st.markdown("### 🪴 Your Notes")

    if os.path.exists("data") and os.listdir("data"):
        files = os.listdir("data")

        for file in files:
            st.markdown(f"""
                <div class="card">
                    <h4>📄 {file}</h4>
                    <p>Stored in your knowledge garden</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No notes yet 🌼")

# ================= ASK AI =================
elif option == "Ask AI":
    st.markdown("### 🤖 Ask Your Knowledge Garden")

    query = st.text_input("Ask something about your notes...")

    if st.button("Get Answer"):
        with st.spinner("Thinking..."):
            engine = index.as_query_engine(llm=llm)
            response = engine.query(query)
            st.success(str(response))

# ================= REFLECTION =================
elif option == "Reflection":
    st.markdown("### 🍁 Reflection")

    if st.button("Generate Reflection"):
        with st.spinner("Analyzing your knowledge..."):
            insight = generate_reflection(index)
            st.success(insight)

# ================= FOOTER =================
st.divider()
st.caption("🌿 Knowledge Garden • Beautiful UI Version")