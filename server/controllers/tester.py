import pinecone

# Initialize Pinecone
api_key = "your-pinecone-api-key"  # Replace with your actual Pinecone API key
environment = "us-west1-gcp"  # Replace with your Pinecone environment

# Initialize the Pinecone client
pinecone.init(api_key=api_key, environment=environment)

# Define index parameters
index_name = "example-index"  # Name of the index
dimension = 128  # Vector dimension size
metric = "cosine"  # Similarity metric ("cosine", "euclidean", or "dotproduct")

# Create the Pinecone index
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=dimension, metric=metric)

# Check if the index was successfully created
print(f"Available indexes: {pinecone.list_indexes()}")
