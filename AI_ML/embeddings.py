import arxiv
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def fetch_papers(query, max_results=10):

    search = arxiv.Search( 
        query = query,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )
 
    papers = [] # list of dictionaries to store each paper's title, summary, and pdf_url
    for result in search.results():
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "pdf_url": result.pdf_url
        })

    return papers

def generate_embeddings(papers):

    model = SentenceTransformer('all-MiniLM-L6-v2') 
    summaries = [paper["summary"]for paper in papers]
    embeddings = model.encode(summaries)
    return embeddings

def save_embeddings(embeddings, papers, index_file="data/papers_index.faiss"):

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    faiss.write_index(index, index_file)
    

if __name__ == "__main__":
  
    query = "AI in healthcare"
    papers = fetch_papers(query)
    embeddings = generate_embeddings(papers)
    save_embeddings(embeddings, papers)

    print("\n Embeddings generated and saved to data/papers_index.faiss")