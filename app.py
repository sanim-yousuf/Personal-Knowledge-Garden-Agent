import streamlit as st
import os
from core import get_index, llm, ingest_and_index, generate_reflection

# ====================== STREAMLIT CONFIG ======================
st.set_page_config(page_title="Knowledge Garden", page_icon="🌿", layout="wide")
st.title("🌿 Knowledge Garden")
st.caption("Advanced Lifelong Second Brain • Powered by **qwen3:8b**")

# Use cache_resource for the index to keep it persistent across interactions
@st.cache_resource
def load_index_cached():
    return get_index()

index = load_index_cached()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("https://picsum.photos/id/1015/300/200", use_column_width=True)
    st.subheader("Garden Status")
    
    data_path = "data"
    total_files = len(os.listdir(data_path)) if os.path.exists(data_path) else 0
    st.metric("Total Items", total_files)
    st.caption("100% Local • qwen3:8b only")

    if st.button("🔄 Hard Reset Index"):
        st.cache_resource.clear()
        st.rerun()

# ====================== TABS ======================
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📤 Ingest", "🌐 Explore", "💡 Insights"])

with tab1:
    st.header("💬 Chat with your Garden")
    if "chat_history" not in st.session_state:
        # Default greeting for Sanim
        st.session_state.chat_history = [{"role": "assistant", "content": "Hello Sanim! Ask me anything about your knowledge garden."}]
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching garden..."):
                query_engine = index.as_query_engine(llm=llm, streaming=False)
                response = query_engine.query(prompt)
                answer = str(response)
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

with tab2:
    st.header("📤 Ingest New Content")
    st.info("Files uploaded here are indexed immediately and stored in your garden.")
    
    uploaded = st.file_uploader("Drop files here", accept_multiple_files=True)
    
    if uploaded:
        if st.button("Ingest Files"):
            with st.spinner("Processing and embedding..."):
                for file in uploaded:
                    ingest_and_index(file.getbuffer(), file.name, index)
            st.success(f"✅ {len(uploaded)} file(s) added to the garden!")
            # Note: We don't strictly need st.rerun() here because index is updated in-memory
            # but it helps refresh the 'Total Items' metric in the sidebar.
            st.rerun()

with tab3:
    st.header("🌐 Explore Garden")
    if os.path.exists("data") and os.listdir("data"):
        files = os.listdir("data")
        st.write(f"**Total items stored**: {len(files)}")
        for f in sorted(files):
            st.write(f"📄 {f}")
    else:
        st.info("Garden is empty. Ingest some files first!")

with tab4:
    st.header("💡 Insights & Reflections")
    st.write("Let qwen3:8b analyze your collected knowledge to find patterns.")
    if st.button("Generate Reflection", type="primary"):
        with st.spinner("Connecting the dots..."):
            insight = generate_reflection(index)
            st.success("Reflection Complete")
            st.markdown(f"> {insight}")

st.divider()
st.caption("Knowledge Garden v2.3 • qwen3:8b • 100% Local & Private")