import streamlit as st
import os
from step3_search import search_lost_item

st.set_page_config(page_title="Urban Asset Recovery", layout="wide")

st.title("🏙️ Urban Asset Recovery: AI Prototype")

# 1. Sidebar for File Management
st.sidebar.header("Data Management")
if st.sidebar.button("Run Indexing"):
    st.sidebar.info("Indexing is already handled by step2_indexing.py")

# 2. Upload the Lost Item
st.header("🔍 Retrieval Search")
query_file = st.file_uploader("Upload Image of Lost Object", type=["jpg", "png", "jpeg"])

if query_file:
    # Save the uploaded file temporarily to search it
    with open("temp_query.jpg", "wb") as f:
        f.write(query_file.getbuffer())
    
    st.image("temp_query.jpg", caption="Query Object", width=300)

    # 3. The Search Button
    if st.button("Find in Surveillance Logs"):
        with st.spinner("Searching vectors..."):
            match_path = search_lost_item("temp_query.jpg")
            
            if match_path:
                st.success("Target Identified in Video Log!")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(match_path, caption="Found Frame")
                with col2:
                    st.metric("Confidence Score", f"{0.85 * 100}%") # Placeholder score
                    st.write(f"Timestamp: {match_path.split('_')[-1].replace('.jpg', '')} seconds")
            else:
                st.error("No match found in current logs.")