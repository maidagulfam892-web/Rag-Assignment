import os
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Maida Gulfam - Digital Clone", layout="wide")

st.title("🤖 Maida Gulfam - Digital Clone (Hybrid RAG)")
st.warning("⚠️ This is an AI clone, not the real person.")

# 1. Knowledge Base Chunks
chunks = [
    {"id": "bio_1", "source": "bio.md", "text": "I am Maida Gulfam, a 7th-semester Computer Science student at UMT Sialkot. I have technical expertise in mobile app development, Python, AI, machine learning, and database administration."},
    {"id": "projects_1", "source": "projects.md", "text": "I built a Node.js REST API integrated with a Flutter mobile application using GetX reactive state management."},
    {"id": "projects_2", "source": "projects.md", "text": "I implemented a custom Role-Based Access Control (RBAC) security mechanism on a university database using SQL Server Management Studio."},
    {"id": "opinions_1", "source": "opinions.md", "text": "I believe mastering backend AI engineering and clean database architecture is essential. In Canva design, I focus on modern visual concepts like my 'Eclipsed' poster series."},
    {"id": "faq_1", "source": "faq.md", "text": "Q: What is your educational background? A: I am a 7th-semester Computer Science student at UMT Sialkot Campus."},
    {"id": "faq_2", "source": "faq.md", "text": "Q: What tech stack do you use? A: My stack includes Flutter, Android Studio, Python, VS Code, Git, Node.js, Kali Linux, and SQL Server."}
]

# 2. Retrievers Initialization
tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_corpus)

model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True)

def search_bm25(query, top_k=3):
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"chunk": chunks[i], "score": float(scores[i]), "id": chunks[i]["id"]} for i in top_indices]

def search_dense(query, top_k=3):
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(chunk_embeddings, q_emb)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"chunk": chunks[i], "score": float(scores[i]), "id": chunks[i]["id"]} for i in top_indices]

def hybrid_rrf(query, top_k=3):
    bm25_res = search_bm25(query, top_k=len(chunks))
    dense_res = search_dense(query, top_k=len(chunks))
    
    rrf_scores = {}
    chunk_map = {c["id"]: c for c in chunks}
    
    for rank, item in enumerate(bm25_res):
        rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0.0) + 1.0 / (60.0 + (rank + 1))
    for rank, item in enumerate(dense_res):
        rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0.0) + 1.0 / (60.0 + (rank + 1))
        
    sorted_fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"chunk": chunk_map[cid], "score": score, "id": cid} for cid, score in sorted_fused]

# UI Layout
query = st.text_area("Ask a question about my background, projects, or studies:")
mode = st.selectbox("Retrieval Mode:", ["Hybrid", "BM25 only", "Dense only"])
hf_token = st.text_input("Hugging Face Token (Optional for LLM):", type="password")

if st.button("Chat with Clone"):
    if not query:
        st.warning("Please enter a question!")
    else:
        if mode == "BM25 only":
            retrieved = search_bm25(query)
        elif mode == "Dense only":
            retrieved = search_dense(query)
        else:
            retrieved = hybrid_rrf(query)
            
        context_str = "\n\n".join([f"Source: {item['chunk']['source']} (ID: {item['id']})\n{item['chunk']['text']}" for item in retrieved])
        
        # LLM Generation
        answer = ""
        client = InferenceClient(model="meta-llama/Llama-3.2-3B-Instruct", token=hf_token) if hf_token else None
        
        if client:
            prompt = f"You are the AI clone of Maida Gulfam, a 7th-semester CS student at UMT Sialkot. Speak in the first person.\nRules: Answer ONLY using the context. If unknown, say 'I don't have that in my notes.'\n\nCONTEXT:\n{context_str}\n\nQUESTION:\n{query}"
            try:
                response = client.chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=200)
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"API Error: {str(e)}. Fallback: {retrieved[0]['chunk']['text']}"
        else:
            answer = f"As a 7th-semester Computer Science student at UMT Sialkot, based on my notes: {retrieved[0]['chunk']['text']}"
            
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Clone Answer")
            st.write(answer)
        with col2:
            st.subheader("Retrieved Chunks & Sources")
            for r in retrieved:
                st.info(f"**Source:** {r['chunk']['source']} | **ID:** {r['id']} | **Score:** {r['score']:.4f}\n\n{r['chunk']['text']}")