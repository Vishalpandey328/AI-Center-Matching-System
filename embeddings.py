from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(text_list):
    """
    Convert text list into embedding vectors
    """

    embeddings = model.encode(text_list)

    return embeddings