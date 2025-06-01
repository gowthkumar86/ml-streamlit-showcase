import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import html
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def render_wikipedia_topic_clusterer_tab():
    st.markdown("""
    <div style='padding:15px; border-radius:10px; margin-bottom:20px; background-color:#f9f9f9;'>
        <h3>Wikipedia Topic Clusterer</h3>
        <p style='font-size:16px;'>
            This application clusters Wikipedia articles based on their content using a Sentence Transformer model.
            It allows users to input a list of Wikipedia article title, fetch their content, and cluster them into topics and key concepts.
            The clustering is performed using <strong>hierarchical clustering</strong> based on cosine distances between the article embeddings.
        </p>
        <p style='font-size:16px;'> For performance reasons, the maximum number of clusters is set to 50.
    </div>
    """, unsafe_allow_html=True)

    # Input for Wikipedia article titles
    title_input = st.text_input("Enter Wikipedia Article Title:","Hierarchical clustering")

    if st.button("Fetch Wikipedia Article"):
        with st.spinner("Wait for it...", show_time=True):
            if title_input:
                url = f"https://en.wikipedia.org/wiki/{title_input.replace(' ', '_')}"
                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    main_content = clean_main_content(soup)
                    if main_content:
                        cleaned_text = clean_wikipedia_text(main_content.get_text(separator="\n", strip=True))
                        sentences = split_sentences(cleaned_text)
                        clusters = hierarchical_cluster_sentences(sentences)
                        df = format_clusters_as_df(clusters)

                        st.subheader("Clustered Topics")
                        st.markdown(f"**Number of Clusters:** {len(clusters)}")

                        for cluster_title, sents in clusters.items():
                            with st.expander(f"{cluster_title} ({len(sents)} sentences)"):
                                for sent in sents:
                                    st.write(f"- {sent}")
                        
                        st.subheader("Clusters DataFrame")
                        st.dataframe(df) 
                else:
                    st.error(response.status_code)




def clean_main_content(soup):
    main_content = soup.select_one('div#bodyContent')
    if main_content:
        infobox = main_content.select_one("table.infobox.vcard.plainlist")
        citation_box1 = main_content.select_one('.box-More_citations_needed.plainlinks.metadata.ambox.ambox-content.ambox-Refimprove')
        if citation_box1:
            citation_box1.decompose()
        if infobox:
            infobox.decompose()
        if main_content.find_all('figure'):
            for tag in soup.find_all('figure'):
                tag.decompose()
        see_also_header = main_content.select_one("h2#See_also")
        notes_header = main_content.select_one("h2#Notes")
        references_header = main_content.select_one("h2#References")
        if see_also_header:
            parent_tag = see_also_header.parent
        elif notes_header:
            parent_tag = notes_header.parent
        elif references_header:
            parent_tag = references_header.parent
        if parent_tag:
            for sibling in list(parent_tag.find_next_siblings()):
                sibling.decompose()
    return main_content


def clean_wikipedia_text(text: str) -> str:
    # Decode HTML entities (e.g., &nbsp;, &#x27;)
    text = html.unescape(text)

    # 1. Remove citation brackets like [51], [edit], etc.
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)

    # # 2. Remove mathematical/LaTeX-like expressions or weird equations
    text = re.sub(r"\{\\displaystyle.*?\}", "", text, flags=re.DOTALL)
    text = re.sub(r"[{}^_\\]+", "", text)

    # 3. Remove sections starting with “See also”, “References”, “Notes”, “External links”
    section_markers = ["See also", "References", "Notes", "External links"]
    
    for marker in section_markers:
        text = text.replace(marker,"")
    text = re.sub(r"\[\s*edit\s*\]", "", text, flags=re.IGNORECASE).strip()

    # 4. Remove “Retrieved from...” and category listings
    text = re.split(r"Retrieved from .*", text)[0]
    text = re.sub(r"Categories?:.*", "", text, flags=re.DOTALL)

    # 5. Remove any remaining edit links or source links
    text = re.sub(r"\[.*?\]", "", text)  # Generic cleanup for leftover square brackets

    # 6. Normalize unicode quotes and dashes
    text = text.replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-")

    # 7. Remove extra whitespace, multiple line breaks, etc.
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove newlines and collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove boilerplate phrases (you can add more if needed)
    patterns_to_remove = [
        r"From Wikipedia, the free encyclopedia",
        r"This article.*?verification\s*\.",  # matches citation warnings
        r"Please help.*?reliable sources\s*\.",
        r"Unsourced material.*?removed\s*\.",
        r"Find sources:.*?\)",  # matches source listings like JSTOR etc.
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove numbered citations like [1], [2], etc.
    text = re.sub(r"\[\s*\d+\s*\]", "", text)

    # Strip leading/trailing whitespace
    return text.strip()

def split_sentences(text):
    # Replace common abbreviations temporarily to avoid splitting there
    abbreviations = {"e.g.": "eg_placeholder", "i.e.": "ie_placeholder"}
    
    for abbr, placeholder in abbreviations.items():
        text = text.replace(abbr, placeholder)
    
    # Now split by period followed by space/newline
    sentences = re.split(r"\.\s+", text)
    
    # Restore the abbreviations
    sentences = [s.replace("eg_placeholder", "e.g.").replace("ie_placeholder", "i.e.") for s in sentences]
    
    return [s.strip() for s in sentences if s.strip()]

def hierarchical_cluster_sentences(sentences, distance_range=(0.5, 3.0, 20)):
    model = load_model() 
    embeddings = model.encode(sentences)

    Z = linkage(embeddings, method='ward')

    # Find optimal `t` based on silhouette score
    best_score = -1
    best_t = None
    best_labels = None

    max_clusters = 50  # set maximum clusters allowed

    for t in np.linspace(*distance_range):
        labels = fcluster(Z, t, criterion='distance')
        n_clusters = len(set(labels))
        n_samples = len(sentences)

        # Skip if less than 2 clusters or more than max_clusters or invalid silhouette condition
        if n_clusters < 2 or n_clusters > max_clusters or n_clusters >= n_samples:
            continue

        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_t = t
            best_labels = labels

    # Group sentences and embeddings by cluster label
    clusters = {}
    for idx, (label, sent) in enumerate(zip(best_labels, sentences)):
        clusters.setdefault(label, []).append((sent, embeddings[idx]))

    named_clusters = {}
    for label, sentence_tuples in clusters.items():
        sents, embs = zip(*sentence_tuples)

        # Calculate centroid embedding for the cluster
        centroid = np.mean(embs, axis=0, keepdims=True)

        # Find the sentence closest to centroid (central sentence)
        distances = cosine_distances(centroid, embs)[0]
        central_idx = np.argmin(distances)
        central_sentence = sents[central_idx]

        # Extract top 3 TF-IDF keywords for the cluster sentences
        sents = [s for s in sents if s.strip()]  # Remove empty strings
        if not sents:
            keywords_str = "No keywords"
        else:
            vectorizer = TfidfVectorizer(stop_words='english')
            try:
                X = vectorizer.fit_transform(sents)
                feature_array = np.array(vectorizer.get_feature_names_out())
                tfidf_scores = X.mean(axis=0).A1
                if len(feature_array) >= 3:
                    top_keywords = feature_array[np.argsort(tfidf_scores)[-3:]][::-1]
                else:
                    top_keywords = feature_array[np.argsort(tfidf_scores)][::-1]

                # Check if any of the top keywords are in the central sentence
                central_sentence_lower = central_sentence.lower()
                keywords_in_central = [kw for kw in top_keywords if kw.lower() in central_sentence_lower]
                if keywords_in_central:
                    keywords_str = ", ".join(keywords_in_central)
                else:
                    keywords_str = ", ".join(top_keywords)
            except ValueError:
                keywords_str = "No keywords"

        cluster_title = f"{keywords_str}"

        named_clusters[cluster_title] = sents

    return named_clusters

def format_clusters_as_df(named_clusters):
    rows = []
    for cluster_title, sents in named_clusters.items():
        for sent in sents:
            rows.append({"Key ideas": cluster_title, "Sentence": sent})
    return pd.DataFrame(rows)