import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#no NVIDIA CUDA Devices, so it works with CPU
#os.environ["TRANSFORMERS_OFFLINE"] = "1"
#os.environ["HF_DATASETS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from search_utils import load_movies, cache_path, create_subfolder

import numpy as np

# Load the model (downloads automatically the first time)


def verify_model():
    semantic_search = SemanticSearch()
    #model.encode(text)
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def test():
    #verify_model()
    semantic_search = SemanticSearch()
    print(semantic_search.generate_embedding("Hello my Friend!"))
    print(semantic_search.generate_embedding("Hello!"))

def embed_text(text):
    semantic_search = SemanticSearch()
    embedded_text = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedded_text[:3]}")
    print(f"Dimensions: {embedded_text.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    movie_list = load_movies()
    semantic_search.load_or_create_embeddings(movie_list)
    
    print(f"Number of docs:   {len(movie_list)}")
    print(f"Embeddings shape: {semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions")


class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

        #embedings file
        self.embeddings_path = cache_path / "movie_embeddings.npy"

    def generate_embedding(self, text):
        if len(text) == 0 or (not text):
            raise ValueError("Text is empty, please try again")
        else:
            text_list = []
            text_list.append(text)
            print(text_list)
            model_result = self.model.encode(text_list)
            #print("#-0.035")

    def build_embeddings(self, documents:list):
        self.documents = documents
        document_list = []
        for document in documents:
        # this is working with the given movie-dictionaries
            self.document_map[document["id"]] = document
            document_string = f"{document['title']}: {document['description']}"
            document_list.append(document_string)
        self.embeddings = self.model.encode(document_list, show_progress_bar=True, batch_size=64)
        
        #check cache-dir and save
        create_subfolder("cache")
        np.save(self.embeddings_path, self.embeddings)
        
        return self.embeddings
    
    def load_or_create_embeddings(self, documents:list):
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document

        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        
        return self.build_embeddings(self.documents)


"""
        print("Building index and Docmap")
        movie_data = load_movies()
        # movie_data is a list of dicts{'id', 'title', 'description'}
        for movie in movie_data:
            doc_id = movie['id']
            movie_text = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.add_document(doc_id, movie_text)
"""