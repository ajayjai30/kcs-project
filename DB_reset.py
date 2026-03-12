from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient("http://localhost:6333")
DOCS_COLLECTION = "app_rag_docs"
REGISTRY_COLLECTION = "app_rag_file_registry"

# --- 1. DELETE COLLECTIONS ---
print(f"🗑️ Deleting collection: {DOCS_COLLECTION}...")
try:
    client.delete_collection(DOCS_COLLECTION)
except Exception:
    pass # Ignores error if it doesn't exist yet

print(f"🗑️ Deleting collection: {REGISTRY_COLLECTION}...")
try:
    client.delete_collection(REGISTRY_COLLECTION)
except Exception:
    pass

# --- 2. RECREATE DOCS (HYBRID) ---
print(f"🏗️ Recreating {DOCS_COLLECTION} (Hybrid Config)...")
client.create_collection(
    collection_name=DOCS_COLLECTION,
    vectors_config=models.VectorParams(
        size=768, # Size for nomic-embed-text:v1.5
        distance=models.Distance.COSINE
    ),
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        )
    }
)

# --- 3. RECREATE REGISTRY ---
print(f"🏗️ Recreating {REGISTRY_COLLECTION} (Registry Config)...")
client.create_collection(
    collection_name=REGISTRY_COLLECTION,
    vectors_config=models.VectorParams(
        size=1, # Size 1 because n8n inserts a dummy [0.0] vector
        distance=models.Distance.COSINE
    )
)

print("✅ Both databases completely wiped and rebuilt! Ready for a fresh start.")