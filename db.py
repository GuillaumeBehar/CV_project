from typing import TypeVar, Union
from chromadb.api.types import Documents, Embeddings, Images, EmbeddingFunction
from sentence_transformers import SentenceTransformer, util
import torch
import chromadb
from PIL import Image
import numpy as np

Embeddable = Union[Documents, Images]
D = TypeVar("D", bound=Embeddable, contravariant=True)


class CustomEmbeddingFunction(EmbeddingFunction[D]):
    def __init__(self):
        self.model = SentenceTransformer('clip-ViT-B-32')

    def __call__(self, input: D) -> Embeddings:
        if not isinstance(input, list):
            input = [input]
        if isinstance(input[0], str):
            with torch.no_grad():
                embeddings = self.model.encode(input)

        else:
            embeddings = self.model.encode(Image.fromarray(np.array(input)))

        return embeddings.tolist()


def store_embeddings(db, image_paths):
    images = [np.array(Image.open(image)).tolist() for image in image_paths]
    db.add(documents=images, ids=image_paths)


def retrieve_embeddings(db, ids):
    embeddings = db.get(ids)
    return embeddings


def query_embeddings(db, query, num_neighbors):
    nearest_neighbors = db.search(query, num_neighbors)
    return nearest_neighbors


if __name__ == "__main__":

    embedding_function = CustomEmbeddingFunction()

    persistent_dir = "./collections"
    client = chromadb.PersistentClient(path=persistent_dir)
    db = client.get_or_create_collection(
        "image_embeddings", embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"})

    image_paths = ["cards/card1.jpg"]
    store_embeddings(db, image_paths)

    embedding_function("rainbow")
    print(query_embeddings(db, "rainbow", 5))
