import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os

# 1. Connect to the existing database
client = chromadb.PersistentClient(path="./recovery_db")
img_embs = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

collection = client.get_or_create_collection(
    name="urban_assets", 
    embedding_function=img_embs,
    data_loader=data_loader
)

def search_lost_item(query_image_path):
    # Ensure the path is absolute
    query_path = os.path.abspath(query_image_path)
    
    print(f"Searching for item: {query_image_path}...")

    # 2. Query the database
    # n_results=1 means we want the single best match
    results = collection.query(
        query_uris=[query_path],
        n_results=1,
        include=["uris", "distances"]
    )

    # 3. Process the results
    if results['uris']:
        matched_uri = results['uris'][0][0]
        distance = results['distances'][0][0]
        print(f"✅ Match Found!")
        print(f"Closest Frame: {os.path.basename(matched_uri)}")
        print(f"Similarity Score: {1 - distance:.4f}") # 1 - distance gives similarity
        return matched_uri
    else:
        print("❌ No match found.")
        return None

if __name__ == "__main__":
    # Test the search with your lost_item.jpg
    search_lost_item("lost_item.jpg")