import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from flask import Flask, jsonify
import threading

# Use fastembed for BM25 computation
try:
    from fastembed import SparseTextEmbedding
except ImportError:
    print("Please install fastembed: pip install fastembed")
    exit(1)

app = Flask(__name__)

# [PRODUCTION UPGRADE] Get Qdrant URL from Docker environment, fallback to localhost for VS Code
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
client = QdrantClient(QDRANT_URL) 
COLLECTION_NAME = "app_rag_docs" #

print("1. Initializing SparseTextEmbedding (BM25)...") #
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25") #

def safe_build_hybrid_index():
    print(f"\n2. Fetching points from Qdrant at {QDRANT_URL}...") 
    points = []
    offset = None
    while True:
        resp, next_page = client.scroll(
            collection_name=COLLECTION_NAME, #
            limit=100, #
            with_payload=True, #
            with_vectors=True, #
            offset=offset #
        )
        points.extend(resp) #
        offset = next_page #
        if offset is None: #
            break

    print(f" -> Found {len(points)} points.") #
    print("\n3. Updating Sparse Vectors safely in batches...") 
    
    batch_size = 50 #
    for i in range(0, len(points), batch_size): #
        batch = points[i : i + batch_size] #
        
        texts = [p.payload.get("content", "") for p in batch] #
        sparse_embeddings = list(sparse_model.embed(texts)) #
        
        updated_points = []
        for p, s_emb in zip(batch, sparse_embeddings): #
            dense_vec = p.vector.get("") if isinstance(p.vector, dict) else p.vector

            vectors_dict = {
                "": dense_vec, 
                "sparse": models.SparseVector( #
                    indices=s_emb.indices.tolist(), #
                    values=s_emb.values.tolist() #
                )
            }
            
            updated_points.append(
                models.PointStruct( #
                    id=p.id, #
                    vector=vectors_dict, #
                    payload=p.payload #
                )
            )
        
        client.upsert(
            collection_name=COLLECTION_NAME, #
            points=updated_points #
        )
        print(f" -> Upserted batch {i//batch_size + 1}") #

    print("\n✅ Safe Hybrid Index update complete!")

@app.route('/trigger-reindex', methods=['POST'])
def trigger_reindex():
    threading.Thread(target=safe_build_hybrid_index).start()
    return jsonify({"status": "Success", "message": "Reindexing started in the background!"}), 200

# [PRODUCTION UPGRADE] This block only runs if you run `python reindex_bm25.py` directly in VS Code.
# Gunicorn will ignore this block in production!
if __name__ == "__main__":
    print("Starting Background Reindex API on port 5000 (Development Mode)...")
    app.run(host='0.0.0.0', port=5000)