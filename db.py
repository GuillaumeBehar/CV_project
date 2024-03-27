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

    def __call__(self, inputs: D) -> Embeddings:
        if not isinstance(inputs, list):
            inputs = [inputs]
        if isinstance(inputs[0], str):
            with torch.no_grad():
                embeddings = self.model.encode(inputs)
            print("Text embeddings shape: ", embeddings.shape)

        else:
            images = [Image.fromarray(np.array(i).astype('uint8'))
                      for i in inputs]
            embeddings = self.model.encode(images)
            print("Image embeddings shape: ", embeddings.shape)

        return embeddings.tolist()


def store_embeddings(db, image_paths):
    images = [np.array(Image.open(image)).tolist() for image in image_paths]
    db.add(documents=images, ids=image_paths)


def get_embeddings(db, ids):
    embeddings = db.get(ids)
    return embeddings


def query_text(db, query, num_neighbors):
    nearest_neighbors = db.query(query_texts=[query], n_results=num_neighbors)
    return nearest_neighbors


def query_image(db, query, num_neighbors):
    image = np.array(Image.open(image_paths[0])).tolist()
    nearest_neighbors = query_text(db, image, 5)
    return nearest_neighbors


if __name__ == "__main__":

    embedding_function = CustomEmbeddingFunction()

    persistent_dir = "./collections"
    client = chromadb.PersistentClient(path=persistent_dir)
    db = client.get_or_create_collection(
        "image_embeddings", embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"})

    image_paths = ["cards/card1.jpg"]
    # store_embeddings(db, image_paths)

    print(query_text(db, "A guy mouting a poney with a rainbow bridge", 5))
