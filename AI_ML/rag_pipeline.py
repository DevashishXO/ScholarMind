# import numpy as np
# import faiss
# import requests
# from sentence_transformers import SentenceTransformer
# import json
# import os

# def load_faiss_index(index_file):

#     index = faiss.read_index(index_file)
#     return index

# def load_papers_metadata(metadata_file="data/papers_metadata.json"):

#     with open(metadata_file, "r") as f:
#         papers = json.load(f)
#     return papers

# def embed_query(query):
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     query_embedding = model.encode([query])
#     return query_embedding

# def search_faiss(index, query_embedding, k =3):
#     _, indices = index.search(query_embedding, k)
#     return indices[0] # returning top k indices

# def get_relevant_summaries(indices, papers):
#     summaries = [papers[i]["summary"] for i in indices]
#     return "\n\n".join(summaries)

# def groq_generate_answer(query, context):
#     API_KEY = os.getenv("GROQ_API_KEY")
#     url = "https://api.groq.com/openai/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json"
#     }
#     data={
#         "model": "llama-3.3-70b-versatile",
#         "messages": [
#             {"role": "system", "content": "You are an AI assistant designed to provide clear and direct answers to user queries. "
#             "When responding, DO NOT mention whether the sources used are relevant or not or if the query is directly addressed with the provided papers. Simply provide a well-structured and user-friendly response, without discussing the context-building process with the user."},
#             {"role": "user", "content": f"Query: {query}\n\nContext: \n{context}\n\nAnswer the query using the given papers as context without discussing the context-building process."}
#         ],
#         "temperature": 0.0,
#     }
#     response = requests.post(url, headers=headers, json=data)
    
#     print("Raw API Response:", response.status_code, response.text)  # Debugging

#     if response.status_code != 200:
#         return "Error: Failed to fetch response from Groq API"
    
#     try:
#         response_json = response.json()
#         return response_json["choices"][0]["message"]["content"]
#     except(KeyError, IndexError):
#         return "Error: Invalid response format from Groq API"

# if __name__ == "__main__":
#     query = input("Enter your research query: ")
    
#     index = load_faiss_index("data/papers_index.faiss")
#     papers = load_papers_metadata()

#     query_embedding = embed_query(query)
#     top_indices = search_faiss(index, query_embedding)
#     context = get_relevant_summaries(top_indices, papers)

#     response = groq_generate_answer(query, context)
#     print("\n AI Response:\n", response)


import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
import requests

MODEL = SentenceTransformer('all-MiniLM-L6-v2')
INDEX_FILE = "data/papers_index.faiss"
METADATA_FILE = "data/papers_metadata.json"

def load_resources():
    """Loading index with error handling"""

    try:
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE ) as f:
            metadata = {str(p['id']): p for p in json.load(f)}
        return index, metadata
    except Exception as e:
        print(f"Error loading resources: {str(e)}")
        return None, None

def find_relevant_context(query, index, metadata, min_relevance = 0.35, min_soft_threshold = 0.32, top_k = 6):
    """Hybrid search with softened relevance threshold, considering both title and summary"""

    id_to_paper = {p['faiss_id']: p for p in metadata.values()}
    query_embedding = MODEL.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding, 10)
    print(f"FAISS returned indices: {indices}")
    print(f"Similarity scores before filtering: {scores}")

    relevant, maybe_relevant = [], []
    for score, idx in zip(scores[0], indices[0]):
        paper = id_to_paper.get(idx)
        if paper:
            if score >= min_relevance:
                relevant.append((score, paper))
            elif score >= min_soft_threshold:
                print(f"Tagging paper '{paper['title']}' as 'maybe relevant' (score: {score})")
                maybe_relevant.append((score, paper))
            else:
                print(f"Rejecting paper '{paper['title']}' due to low relevance score: {score}")

    combined = sorted(relevant, reverse = True)[:top_k]
    remaining_slots = top_k - len(combined)
    if remaining_slots > 0:
        combined += sorted(maybe_relevant, reverse = True)[:remaining_slots]
        
    print(f"Selected papers with scores: {[(s, p['title']) for s, p in combined]}")        
    # print(f"Selected Papers with Scores: {[(s, p['title']) for s, p in relevant]}")

    return combined
    
def generate_grounded_response(query, context_papers):
    """Structured prompting for accuracy."""
 
    context_str = "\n".join(
        f"[Source {i+1}] {p['title']} \n{p['summary'][:300]}"
        + (" (This paper may only be partially relevant.)" if score < 0.35 else "")
        for i, (score, p) in enumerate(context_papers)
    )

    prompt = f"""You are an academic research assistant. Answer the query using the provided context. You MUST use the listed papers as context, even if they are only partially relevant.

Query:
{query}

Context:
{context_str}

Formatting & Writing Guidelines:
1. If unsure, begin with: "Based on current research: [general insight]."
2. Always cite sources using square brackets like [1], [2], etc.
3. When referencing multiple sources, use separate brackets for each (e.g., [1][2][3]), not a comma-separated list like [1, 2, 3].
4. Place citations **at the end of the sentence or clause** — avoid embedding them mid-sentence.
5. For responses involving equations, use LaTeX formatting in a string format which will be sent to the backend for rendering. 
6. Ensure each paper is cited **at least once** where relevant. Do not mention the context section directly.
7. Use clear, well-structured, and technical academic language.

Some papers may be only partially relevant. You can choose to include their insights if helpful, but do not force them in.
Your goal is to deliver a clean, academic-style answer that would be suitable for inclusion in a research report or literature review.
"""


    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},

        json = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
    )

    return response.json()['choices'][0]['message']['content']

def main():
    index, metadata = load_resources()
    if index is None:
        print("Initializing knowledge base...")
        from embeddings import update_knowledge_base
        update_knowledge_base("artificial intelligence")
        index, metadata = load_resources()

    while True:
        query = input("\nResearch Query (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        # Check existing relevance
        results = find_relevant_context(query, index, metadata)

        # Expand knowledge base if needed
        if not results:
            from embeddings import update_knowledge_base
            print("Updating knowledge base with new research...")
            if update_knowledge_base(query):
                index, metadata = load_resources()
                results = find_relevant_context(query, index, metadata)

        # Generate response
        if results:
            response = generate_grounded_response(query, results)
            sources = "\n".join(
                f"{p['title']} - {p['pdf_url']}"
                for _, p in results
            )
            print(f"\nRESPONSE:\n{response}\n\nSources:\n{sources}")

        else:
            print("No relevant papers found after knowledge base update.")


if __name__ == "__main__":
    main()

