import faiss
import numpy as np

def build_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index


def search_index(index, query_embedding, top_k=5):

    distances, indices = index.search(query_embedding, top_k)

    return distances, indices