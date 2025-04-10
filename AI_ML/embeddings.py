import arxiv
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import os
import re

DATA_DIR = "data"
INDEX_FILE = os.path.join(DATA_DIR, "papers_index.faiss")
METADATA_FILE = os.path.join(DATA_DIR, "papers_metadata.json")


THRESHOLD_ACCEPT = 0.27

def initialize_files():
    """Ensure valid initial file structure"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "w") as f:
            json.dump([], f)

def clean_text(text):
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:2000]

def fetch_relevant_papers(query, max_results=150):
    """Advanced arxiv query construction"""

    processed_query = f"ti:({query}) OR abs:({query})"
    category_filters = [
        'cat:cs.AI', 'cat:cs.LG', 'cat:cs.CL', 'cat:stat.ML', 'cat:cs.CV', 'cat:cs.CY', 'cat:cs.HC',
        'cat:physics.gen-ph', 'cat:q-bio.BM', 'cat:stat.AP'
    ]

    search = arxiv.Search(
        query=f"{processed_query} AND ({' OR '.join(category_filters)})",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []

    for result in arxiv.Client().results(search):
        arxiv_id = result.entry_id.split('/')[-1].split('v')[0]
        title = clean_text(result.title)
        summary = clean_text(result.summary)

        paper_data = {
            "id": str(abs(hash(arxiv_id))),
            "faiss_id": abs(hash(arxiv_id)),
            "arxiv_id": arxiv_id,
            "title": title,
            "summary": summary,
            "pdf_url": result.pdf_url
        }
        papers.append(paper_data)

    return papers

def create_optimized_index(embeddings, papers):
    """FAISS index with proper normalization"""
    dimension = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
    
    ids = np.array([p['faiss_id'] for p in papers], dtype=np.int64)
    faiss.normalize_L2(embeddings)
    index.add_with_ids(embeddings, ids) 
    return index

def update_knowledge_base(query):
    initialize_files()
    
    all_papers = fetch_relevant_papers(query)
    if not all_papers:
        print("No new papers found from Arxiv.")
        return False

    try:
        with open(METADATA_FILE, "r") as f:
            existing_papers = json.load(f)
            existing_ids = {p['id'] for p in existing_papers}
    except json.JSONDecodeError:
        existing_papers = []
        existing_ids = set()

    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    accepted = []
    rejected = []
    rejected_scores = []

    for paper in all_papers:
        if paper['id'] in existing_ids:
            continue

        combined = paper["summary"] + " " + paper["title"]
        paper_embedding = model.encode([combined], normalize_embeddings=True)[0]
        score = np.dot(query_embedding, paper_embedding)

        if score >= THRESHOLD_ACCEPT:
            accepted.append(paper)
        else:
            rejected.append(paper["title"])
            rejected_scores.append(score)

    if not accepted:
        print("No relevant new papers to add.")
        return False

    embeddings = model.encode([p["summary"] + " " + p["title"] for p in accepted], normalize_embeddings=True)

    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        ids = np.array([abs(hash(p['id'])) for p in accepted], dtype=np.int64)
        index.add_with_ids(embeddings, ids)
    else:
        index = create_optimized_index(embeddings, accepted)
    
    faiss.write_index(index, INDEX_FILE)

    updated_papers = existing_papers + accepted
    with open(METADATA_FILE, "w") as f:
        json.dump(updated_papers, f, indent=2)


    print(f"✅ Added {len(accepted)} relevant papers.")
    print(f"❌ Rejected {len(rejected)} papers for query: {query}")
    print("Rejected Paper Titles:", rejected[:10])

    if rejected_scores:
        print("🔻 Highest similarity among rejected papers:", round(max(rejected_scores), 4))

    return True


if __name__ == "__main__":
    initialize_files()
    user_query = input("Enter initial research topic: ")
    success = update_knowledge_base(user_query)
    if success:
        print("✅ Knowledge base initialized successfully!")
    else:
        print("⚠️ No new papers were added to the knowledge base.")
