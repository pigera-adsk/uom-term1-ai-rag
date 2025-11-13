import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Any
import uvicorn
import contextlib

# --- Global Variables ---
model: SentenceTransformer = None
index: faiss.Index = None
datastore: List[Dict[str, Any]] = []

# --- Dummy Data (Same as before) ---
DUMMY_DOCS = [
    # ... (no change from original code) ...
    {
        "content": "Welcome to the Electrical Engineering Department admissions page for 2023. We offer programs in power systems and electronics.",
        "metadata": {"department": "Electrical", "year": "2023", "type": "admissions"}
    },
    {
        "content": "The Computer Science Department curriculum for 2022 includes courses on AI, data structures, and software engineering.",
        "metadata": {"department": "Computer Science", "year": "2022", "type": "curriculum"}
    },
    {
        "content": "University of Moratuwa general regulations and student conduct rules, updated for 2023.",
        "metadata": {"department": "General", "year": "2023", "type": "regulation"}
    },
    {
        "content": "Research spotlight: Electrical Engineering 2023 projects on renewable energy and smart grids.",
        "metadata": {"department": "Electrical", "year": "2023", "type": "research"}
    },
    {
        "content": "A guide to library resources for all Computer Science students (2023).",
        "metadata": {"department": "Computer Science", "year": "2023", "type": "guide"}
    },
    {
        "content": "Information on 2023 admissions for postgraduate studies in Computer Science.",
        "metadata": {"department": "Computer Science", "year": "2023", "type": "admissions"}
    }
]

# --- Startup Function (Same as before) ---
def setup_vector_store():
    """
    Builds the FAISS index from DUMMY_DOCS on application startup.
    """
    global model, index, datastore
    print("Setting up vector store...")

    # ... (function content is identical to the original) ...
    
    datastore = DUMMY_DOCS
    contents = [doc['content'] for doc in datastore]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Creating embeddings for documents...")
    embeddings = model.encode(contents, convert_to_tensor=False, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"Index created with {index.ntotal} vectors.")
    print("✅ Vector store setup complete.")


# --- 4. FastAPI Application (UPDATED) ---

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    The new lifespan function. Code before `yield` runs on startup.
    """
    # Load the model and build the index
    setup_vector_store()
    
    # Let the application run
    yield
    
    # Code after `yield` would run on shutdown (e.g., cleanup)
    print("Shutting down...")


app = FastAPI(
    title="University Document Search API",
    description="A Home Lab project using FastAPI and FAISS for metadata-filtered retrieval.",
    lifespan=lifespan,
    docs_url=None,  # <-- This disables the "/docs" (Swagger UI) page
    redoc_url=None  # <-- This disables the "/redoc" (ReDoc) page
)

# <-- 3. REMOVE THE OLD @app.on_event("startup") FUNCTION -->


@app.get("/search", response_model=List[Dict[str, Any]])
async def search(
    query: str,
    department: Optional[str] = Query(None, description="Filter by department"),
    year: Optional[str] = Query(None, description="Filter by year")
):
    """
    Search the vector store with metadata filtering.
    """
    # ... (function content is identical to the original) ...

    global model, index, datastore
    print(f"\nReceived query: '{query}', filters: department={department}, year={year}")

    allowed_ids = []
    for i, doc in enumerate(datastore):
        meta = doc['metadata']
        dept_match = (department is None) or (meta.get('department') == department)
        year_match = (year is None) or (meta.get('year') == year)
        if dept_match and year_match:
            allowed_ids.append(i)
    
    if not allowed_ids:
        print("No documents matched metadata filters.")
        return []
        
    print(f"Filtered down to {len(allowed_ids)} documents: {allowed_ids}")

    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype('float32')
    faiss.normalize_L2(query_vector)

    k = 2
    search_params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed_ids))
    distances, indices = index.search(query_vector, k=k, params=search_params)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        
        doc = datastore[int(idx)]
        results.append({
            "score": float(dist),
            "content": doc['content'],
            "metadata": doc['metadata']
        })

    print(f"Found {len(results)} results.")
    return results


if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=8000)