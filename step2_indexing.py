import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader # Added this
import os

# 1. Setup ChromaDB
client = chromadb.PersistentClient(path="./recovery_db")

# 2. Setup the Brain & the Loader
img_embs = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader() # This tool reads the JPG files

# 3. Create/Get the collection with the loader
collection = client.get_or_create_collection(
    name="urban_assets", 
    embedding_function=img_embs,
    data_loader=data_loader # Tell the collection how to load images
)

def index_frames(folder="frames"):
    # Get the FULL paths to the images
    all_frames = [os.path.abspath(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith('.jpg')]
    
    if not all_frames:
        print("Error: No frames found.")
        return

    print(f"Indexing {len(all_frames)} frames...")
    
    # We use 'uris' instead of 'images' because we are passing file paths
    collection.add(
        ids=[os.path.basename(f) for f in all_frames],
        uris=all_frames 
    )
    print("✅ Success! Your surveillance logs are now a searchable vector database.")

if __name__ == "__main__":
    index_frames()