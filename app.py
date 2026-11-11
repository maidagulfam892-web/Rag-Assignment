import os
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# Page Configuration
st.set_page_config(
    page_title="Maida Gulfam - Digital Clone", 
    page_icon="🤖", 
    layout="wide"
)

# Custom Styling for Clean Chat UI and Table
st.markdown("""
    <style>
    .user-box { background-color: #f8f9fa; padding: 12px 18px; border-radius: 8px; border: 1px solid #e5e7eb; margin-bottom: 10px; font-weight: 500; }
    .answer-box { background-color: #ffffff; padding: 18px; border-radius: 8px; border: 1px solid #e5e7eb; margin-bottom: 15px; font-size: 1.05rem; line-height: 1.6; color: #1f2937; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Maida Gulfam - Digital Clone (Hybrid RAG)")
st.warning("⚠️ This is an AI clone, not the real person.")

# 1. Detailed Knowledge Base Chunks covering complete academic, technical, and lifestyle profile
chunks = [
    {"id": "bio_1", "source": "bio.md", "chunk_idx": "#0", "text": "I am Maida Gulfam, a 7th-semester Computer Science student at UMT Sialkot. I have comprehensive technical expertise in mobile app development, Python, artificial intelligence, machine learning, and database administration."},
    {"id": "bio_2", "source": "bio.md", "chunk_idx": "#1", "text": "My daily lifestyle and academic routine include maintaining a collection of university-wear abayas in zinc, mint green, black, brown, and beige tones. I also regularly use Organic Traveller skincare products and Jenpharm moisturizer."},
    {"id": "projects_1", "source": "projects.md", "chunk_idx": "#0", "text": "I built and integrated a Node.js REST API with a Flutter mobile application using GetX reactive state management, handling user model decoding, load state mapping, and local endpoint records."},
    {"id": "projects_2", "source": "projects.md", "chunk_idx": "#1", "text": "I implemented a custom Role-Based Access Control (RBAC) security mechanism featuring four distinct authorization roles on a university database using SQL Server Management Studio and explicit T-SQL query structures."},
    {"id": "projects_3", "source": "projects.md", "chunk_idx": "#2", "text": "I completed a digital forensics metadata modification and verification lab using ExifTool on Linux, successfully updating image metadata to assign artist name Maida Gulfam and verifying data integrity via SHA-256 hashing logs."},
    {"id": "opinions_1", "source": "opinions.md", "chunk_idx": "#0", "text": "I believe mastering backend AI engineering and clean database architecture is essential. In digital graphic design using Canva, I focus on modern visual concepts like my 'Eclipsed' poster series showcased on @designs.by.maida."},
    {"id": "faq_1", "source": "faq.md", "chunk_idx": "#0", "text": "Q: What is your educational background? A: I am a 7th-semester Computer Science student at UMT Sialkot Campus."},
    {"id": "faq_2", "source": "faq.md", "chunk_idx": "#1", "text": "Q: What tech stack and tools do you use? A: My technical toolchain includes Flutter, Android Studio, Python, VS Code, Git, Node.js, Kali Linux, and SQL Server Management Studio."},
    {"id": "faq_3", "source": "faq.md", "chunk_idx": "#2", "text": "Q: What professional experience and internships do you have? A: I completed internships at Big Brains (working on AI and Flutter tracks) and FlyRank AI focusing on machine learning workflow audits and AI tool integrations."},
    {"id": "faq_4", "source": "faq.md", "chunk_idx": "#3", "text": "Q: How do you relax and unwind? A: I relax by crafting, digital graphic design, and on weekends I enjoy unwinding by watching K-dramas."}
]

# 2. Retrievers Initialization (BM25 and Sentence Transformers)
tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_corpus)

model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True)

def search_bm25(query, top_k=3):
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"chunk": chunks[i], "score": float(scores[i]), "id": chunks[i]["id"], "chunk_idx": chunks[i]["chunk_idx"]} for i in top_indices]

def search_dense(query, top_k=3):
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(chunk_embeddings, q_emb)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"chunk": chunks[i], "score": float(scores[i]), "id": chunks[i]["id"], "chunk_idx": chunks[i]["chunk_idx"]} for i in top_indices]

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
    return [{"chunk": chunk_map[cid], "score": score, "id": cid, "chunk_idx": chunk_map[cid]["chunk_idx"]} for cid, score in sorted_fused]

# UI Layout Components
query = st.text_area("Ask me anything about my projects, research, experience, skills, hobbies...", placeholder="e.g., Tell me about your technical skills and projects.")

col1, col2 = st.columns([2, 2])
with col1:
    mode = st.selectbox("Retrieval Mode:", ["Hybrid", "BM25 only", "Dense only"])
with col2:
    default_token = os.environ.get("HF_TOKEN", "")
    hf_token = st.text_input("Hugging Face Token (Space Secret / Optional):", value=default_token, type="password")

if st.button("Chat with Clone"):
    if not query.strip():
        st.warning("Please enter a question!")
    else:
        with st.spinner("Processing retrieval and generating clone response..."):
            if mode == "BM25 only":
                retrieved = search_bm25(query)
            elif mode == "Dense only":
                retrieved = search_dense(query)
            else:
                retrieved = hybrid_rrf(query)
                
            context_str = "\n\n".join([f"Source: {item['chunk']['source']} (ID: {item['id']})\n{item['chunk']['text']}" for item in retrieved])
            
            # Strict Persona Prompt forcing rich, multi-sentence, first-person synthesis
            answer = ""
            client = InferenceClient(model="meta-llama/Llama-3.2-3B-Instruct", token=hf_token) if hf_token else None
            
            if client:
                prompt = f"""You are the professional AI digital clone of Maida Gulfam, a 7th-semester Computer Science student at UMT Sialkot. 
Your task is to answer the user's question thoroughly, naturally, and directly in the first person ("I"). 
Combine and synthesize all relevant details from the provided context passages into a rich, well-rounded, and conversational response. Do not give a single short sentence if multiple context details apply.

Rules:
1. Answer strictly using ONLY the provided CONTEXT. Do not invent or guess any outside facts.
2. If the context does not contain the answer, say: "I don't have that in my notes."
3. Never disclose phone numbers, addresses, banking details, or sensitive IDs.

CONTEXT:
{context_str}

QUESTION:
{query}

ANSWER:"""
                try:
                    response = client.chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=300)
                    answer = response.choices[0].message.content.strip()
                except Exception as e:
                    answer = f"API Error: {str(e)}. Fallback: {retrieved[0]['chunk']['text']}"
            else:
                answer = f"As a 7th-semester Computer Science student at UMT Sialkot, based on my notes: {retrieved[0]['chunk']['text']}"
                
        # Render clean chat response cards
        st.markdown(f'<div class="user-box">👤 {query}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">🌸 {answer}</div>', unsafe_allow_html=True)
        
        # Exact Matching Expandable Table Panel for Retrieved Knowledge Sources & Scores
        with st.expander("🔍 Retrieved Knowledge Sources & Scores", expanded=True):
            table_data = []
            for idx, r in enumerate(retrieved, start=1):
                table_data.append({
                    "Rank": f"#{idx}",
                    "Source File": r['chunk']['source'],
                    "Chunk ID": r['chunk_idx'],
                    "RRF Score": f"{r['score']:.4f}",
                    "Passage Snippet": r['chunk']['text']
                })
            st.table(table_data)