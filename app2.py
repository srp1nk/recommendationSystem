import streamlit as st
import torch
from sentence_transformers import util
from collections import defaultdict
import pandas as pd
from sentence_transformers import SentenceTransformer

# ✅ Важно: должен быть ВПЕРВЫЕ вызванным Streamlit-командой
st.set_page_config(page_title="Article Search & Author Recommendation", layout="wide")

# Теперь можно кэшировать
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

@st.cache_data
def generate_embeddings(dataset):
    abstracts = dataset['abstract'].astype(str).tolist()
    return model.encode(abstracts, convert_to_tensor=True)
    
nlp_dataset = pd.read_csv("nlp_dataset.csv").reset_index(drop=True)

embeddings = generate_embeddings(nlp_dataset)


available_topics = ["All topics"] + sorted(nlp_dataset['topic_label'].dropna().unique().tolist())

st.title("🔎 Semantic Article Search & Author Recommendation")

tab1, tab2 = st.tabs(["Search Articles", "Recommend Authors"])

with tab1:
    query = st.text_input("🔍 Enter your query:")
    selected_topic = st.selectbox("📂 Choose a topic (optional):", options=available_topics)
    if st.button("Search for Articles"):
        if not query.strip():
            st.warning("⚠️ Please enter a query.")
        else:
            query_embedding = model.encode(query, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
            top_k = 5
            top_results = torch.topk(cosine_scores, k=top_k)
            top_indices = top_results.indices.tolist()
            top_scores = top_results.values.tolist()

            if selected_topic != "All topics":
                filtered = [(i, s) for i, s in zip(top_indices, top_scores)
                            if nlp_dataset.iloc[i]['topic_label'] == selected_topic]
            else:
                filtered = list(zip(top_indices, top_scores))

            if not filtered:
                st.info("😕 There are no suitable articles on the chosen topic.")
            else:
                st.subheader(f"🔍 Results for: \"{query}\"")
                if selected_topic != "All topics":
                    st.write(f"📂 Topic: {selected_topic}")
                st.write(f"🔢 Relevant articles found: {len(filtered)}")
                st.markdown("---")

                author_scores = defaultdict(float)

                for i, score in filtered[:top_k]:
                    row = nlp_dataset.iloc[i]
                    st.markdown(f"📍 **Link:** {row['id']}")
                    st.markdown(f"📘 **{row['title']}** (Score: {score:.4f})")
                    st.markdown(f"📂 Topic: {row['topic_label']}")
                    st.markdown(f"📝 Abstract: {row['abstract'][:300]}...\n")

                    authors = row['authors']
                    if isinstance(authors, str):
                        try:
                            parsed = eval(authors)
                            if isinstance(parsed, list):
                                authors = [str(a).strip() for a in parsed]
                            else:
                                authors = [a.strip() for a in authors.split(',')]
                        except:
                            authors = [a.strip() for a in authors.split(',')]
                    elif isinstance(authors, list):
                        authors = [str(a).strip() for a in authors]
                    else:
                        authors = []

                    for author in authors:
                        if author:
                            author_scores[author] += float(score)

                st.markdown("---")
                st.markdown("👥 **Top Recommended Authors:**")
                sorted_authors = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)
                for i, (author, total_score) in enumerate(sorted_authors[:5], 1):
                    st.write(f"{i}. {author} (Relevance Score: {total_score:.4f})")

with tab2:
    author_query = st.text_input("🧠 Enter a topic or question:")
    if st.button("Recommend Authors"):
        if not author_query.strip():
            st.warning("⚠️ Please enter a query.")
        else:
            query_embedding = model.encode(author_query, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
            top_k = 5
            top_results = torch.topk(cosine_scores, k=top_k)
            top_indices = top_results.indices.tolist()
            top_scores = top_results.values.tolist()

            author_scores = defaultdict(float)

            for idx, score in zip(top_indices, top_scores):
                row = nlp_dataset.iloc[idx]
                authors = row['authors']

                if isinstance(authors, str):
                    try:
                        parsed = eval(authors)
                        if isinstance(parsed, list):
                            authors = [str(a).strip() for a in parsed]
                        else:
                            authors = [a.strip() for a in authors.split(',')]
                    except:
                        authors = [a.strip() for a in authors.split(',')]
                elif isinstance(authors, list):
                    authors = [str(a).strip() for a in authors]
                else:
                    authors = []

                for author in authors:
                    if author:
                        author_scores[author] += float(score)

            if not author_scores:
                st.info("😕 No authors found for this query.")
            else:
                st.subheader(f"👥 Top Recommended Authors for: \"{author_query}\"")
                st.markdown("---")
                sorted_authors = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)
                for i, (author, score) in enumerate(sorted_authors[:5], 1):
                    st.write(f"{i}. {author} (Relevance Score: {score:.4f})")