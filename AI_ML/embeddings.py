# import arxiv
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss
# import json
# import os

# def fetch_papers(query, max_results=10):

#     search = arxiv.Search( 
#         query = query,
#         max_results = max_results,
#         sort_by = arxiv.SortCriterion.SubmittedDate
#     )
 
#     papers = [] 
#     for result in search.results():
#         papers.append({
#             "title": result.title,
#             "summary": result.summary,
#             "pdf_url": result.pdf_url
#         })

#     return papers

# def generate_embeddings(papers):

#     model = SentenceTransformer('all-MiniLM-L6-v2') 
#     summaries = [paper["summary"]for paper in papers]
#     embeddings = model.encode(summaries)
#     return embeddings

# def save_embeddings(embeddings, papers, index_file="data/papers_index.faiss", metadata_file="data/papers_metadata.json"):

#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(np.array(embeddings))
#     faiss.write_index(index, index_file)

#     indexed_papers = [{"id": i, **paper} for i, paper in enumerate(papers)]
#     with open(metadata_file, "w") as f:
#         json.dump(indexed_papers, f)

# if __name__ == "__main__":
  
#     query = "AI in healthcare"
#     papers = fetch_papers(query)
#     embeddings = generate_embeddings(papers)
#     save_embeddings(embeddings, papers)

#     print("\n Embeddings and papers generated and saved to data/papers_index.faiss and data/papers_metadata.json")


# import arxiv
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss
# import json
# import os

# DATA_DIR = "data"
# INDEX_FILE = os.path.join(DATA_DIR, "papers_index.faiss")
# METADATA_FILE = os.path.join(DATA_DIR, "papers_metadata.json")

# PRELOADED_DOMAINS=[
#     "AI in healthcare",
#     "AI in education",
#     "Natural Language Processing",
#     "Computer Vision in Medicine",
#     "AI in finance"
# ]

# def fetch_papers(query, max_results = 50, preload = False):
#     """Fetching papers from arxiv based on a query"""

#     search = arxiv.Search(
#         query = query,
#         max_results=max_results,
#         sort_by=arxiv.SortCriterion.SubmittedDate
#     )
#     client = arxiv.Client()
#     results = client.results(search) 

#     papers = []
#     for result in results:
#         papers.append({
#             "title": result.title,
#             "summary": result.summary,
#             "pdf_url": result.pdf_url
#         })

#     if preload:
#         print(f"[Preloading] {len(papers)} papers fetched for query: '{query}'")
#     else:
#         print(f"[Dynamically fetching] {len(papers)} papers fetched for query: '{query}'")

#     return papers

# def generate_embeddings(papers):
#     """Generating sentence embeddings for paper summaries"""

#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     summaries = [paper["summary"] for paper in papers]
#     embeddings = model.encode(summaries)
#     return embeddings

# def save_embeddings(embeddings, papers, index_file=INDEX_FILE, metadata_file=METADATA_FILE):
#     """Saving embeddings to FAISS index and storing metadata"""

#     # Load existing metadata if available
#     indexed_papers = []
#     if os.path.exists(metadata_file):
#         with open(metadata_file, "r") as f:
#             indexed_papers = json.load(f)

#     # Track existing titles to avoid duplication
#     existing_titles = set(paper["title"] for paper in indexed_papers)

#     # Filter out duplicate papers
#     new_papers = [paper for paper in papers if paper["title"] not in existing_titles]
#     if not new_papers:
#         print("[Info] No new papers to add.")
#         return

#     new_embeddings = generate_embeddings(new_papers)  # Generate embeddings for new papers

#     # Load FAISS index
#     if os.path.exists(index_file):
#         index = faiss.read_index(index_file)
#     else:
#         dimension = new_embeddings.shape[1]
#         index = faiss.IndexFlatIP(dimension)
#         faiss.normalize_L2(new_embeddings)

#     # Add only new embeddings
#     index.add(np.array(new_embeddings))
#     faiss.write_index(index, index_file)

#     # Assign unique IDs and store new metadata
#     start_id = len(indexed_papers)
#     for i, paper in enumerate(new_papers):
#         paper["id"] = start_id + i
#         indexed_papers.append(paper)

#     with open(metadata_file, "w") as f:
#         json.dump(indexed_papers, f, indent=4)

#     print(f"[Saved] {len(new_papers)} new papers added. Total: {len(indexed_papers)}")


# def load_index(index_file=INDEX_FILE, metadata_file=METADATA_FILE):  
#     """Loading FAISS index and metadata"""

#     if not os.path.exists(index_file) or not os.path.exists(metadata_file):
#         print("[Error] FAISS index or metadata file not found. Please generate embeddings first.")
#         return None, []
    
#     index = faiss.read_index(index_file)
#     print(f"🔍 FAISS index contains {index.ntotal} stored papers.")
#     with open(metadata_file, "r") as f:
#         papers = json.load(f)

#     return index, papers

# def find_relevant_papers(query_embedding, index, stored_papers, k=3):
#     faiss.normalize_L2(query_embedding)  # Normalize query
#     distances, indices = index.search(query_embedding, k) # find k nearest neighbors
    
#     for i, score in zip(indices[0], distances[0]):
#         if i < len(stored_papers):
#             print(f"Matched: {stored_papers[i]['title']} (Score: {score:.4f})")
    
#     relevant_papers = [stored_papers[i] for i in indices[0] if i < len(stored_papers)]
#     return relevant_papers



# if __name__ == "__main__":
#     import sys

#     # Loading stored index and metadata
#     index, stored_papers = load_index()
#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     # Checking if running preloading mode
#     if len(sys.argv) > 1 and sys.argv[1] == "preload":
#         print("\n[Preloading Mode] Clearing existing FAISS index and metadata...")
#         if os.path.exists(INDEX_FILE):
#             os.remove(INDEX_FILE)
#         if os.path.exists(METADATA_FILE):
#             os.remove(METADATA_FILE)

#         for domain in PRELOADED_DOMAINS:
#             papers = fetch_papers(domain, preload=True)
#             embeddings = generate_embeddings(papers)
#             save_embeddings(embeddings, papers)
#         print("\n✅ Preloading completed.")
#         sys.exit()

#     # If query being provided via CLI or prompt
#     query = input("Enter your research query: ") if len(sys.argv) == 1 else " ".join(sys.argv[1:])

#     query_embedding = model.encode([query])
#     relevant_papers = find_relevant_papers(query_embedding, index, stored_papers)

#     if relevant_papers:
#         print(f"\n🔍 Found {len(relevant_papers)} relevant papers from stored embeddings.")

#     else:
#         print("\n⚠️ No relevant papers found in stored embeddings. Fetching dynamically from Arxiv...")
#         papers = fetch_papers(query)
#         embeddings = generate_embeddings(papers)
#         save_embeddings(embeddings, papers)
#         relevant_papers = papers

#     # Outputting relevant papers
#     for i, paper in enumerate(relevant_papers, 1):
#         print(f"\n[{i}] {paper['title']}\n🔗 {paper['pdf_url']}\n📄 {paper['summary'][:300]}...")


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
