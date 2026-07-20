import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#no NVIDIA CUDA Devices, so it works with CPU
#os.environ["TRANSFORMERS_OFFLINE"] = "1"
#os.environ["HF_DATASETS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from search_utils import load_movies, cache_path, create_subfolder

import numpy as np

# Load the model (downloads automatically the first time)

offline_mode = 1 #os.getenv("OFFLINE_MODE") == "1"

#####################################################################
## CLI Methods
#####################################################################

def search_command(query, limit=5):
    #print("Hellooo")
    #print("LIMIT:", limit)
    semantic_search = SemanticSearch()
    movie_list = load_movies()
    semantic_search.load_or_create_embeddings(movie_list)
    results = semantic_search.search(query, limit)
    for i, result in enumerate(results):
        #print(f"{i+1}. {result["title"]} (score: {result["score"]:.4f}) \n {result["description"]}\n\n")
        print(f"{i+1}. {result["title"]} (score: {result["score"]:.4f}) \n")

def chunk_command(text, chunk_size, overlap):
    print(chunk_size)
    text_list = text.split()
    chunk_list = []
    chunk_string = ""
    i = 0
    print(text_list)
    while i <= len(text_list):
        if i==0:
            start = 0
            end = start+chunk_size
            #end = start+chunk_size+overlap
            #print(f"start: {start}, end: {end}")
        if i>0:
            start = i-overlap
            if start < 0: start = 0
            end = start+chunk_size+overlap
            if end > len(text_list): end = len(text_list)
            #print(f"start: {start}, end: {end}")
        #chunk = text_list[start:end]
        #chunk = " ".join(chunk)
        chunk = " ".join(text_list[start:end])
        #print(f"chunk: {chunk}")
        if chunk: 
            chunk_list.append(chunk)
        print(i)
        i = i + chunk_size

        return text_list
        
def semantic_chunk_command(text, max_chunk_size, overlap):
    import re
    print("SEMANTIC CHUNKING")
    chunk_list = []
    text_list = re.split(r"(?<=[.!?])\s+",text)
    #print(text_list)
    chunk_size = max_chunk_size
    chunk_string = ""
    
    #schleife
    start = 0
    end = start + chunk_size
    if overlap >= max_chunk_size:
        overlap = max_chunk_size - 1
        print(f"Overlap has been overwritten to {overlap}")
        
    while start <= len(text_list):
        chunk = " ".join(text_list[start:end])
        if chunk: chunk_list.append(chunk)
        if end >= len(text_list): break
        start = end - overlap #ende ist exklusiv, deshalb ist ende das nächste wort - minus overlap
        end = start + chunk_size        
    
    #print(chunk_list)
    #print("\n")
    # Output text:
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunk_list):
        print(f"{i+1}. {chunk}")
    
    return chunk_list
#####################################################################
## Working Methods
#####################################################################
def embed_text(text):
    semantic_search = SemanticSearch()
    embedded_text = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedded_text[:3]}")
    print(f"Dimensions: {embedded_text.shape[0]}")

def embed_query(query):
    semantic_search = SemanticSearch()
    embedded_query = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedded_query[:3]}")
    print(f"Shape: {embedded_query.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    #a fast way to calculate the cosine similarity of two vectors (1.0 -> same direction, 0.0 -> unrelated, -1.0 -> opposite direction)
    #score = np.dot(query_embedding, doc_embedding)
    #score = np.dot(query_embedding, embedding)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

#####################################################################
## Verify / Testing
#####################################################################

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

def verify_embeddings():
    semantic_search = SemanticSearch()
    movie_list = load_movies()
    semantic_search.load_or_create_embeddings(movie_list)
    
    print(f"Number of docs:   {len(movie_list)}")
    print(f"Embeddings shape: {semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions")


#####################################################################
## Working Class (haha)
#####################################################################

class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=offline_mode)#, device="cpu")
        #model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

        #embedings file
        self.embeddings_path = cache_path / "movie_embeddings.npy"

    def generate_embedding(self, text):
        if not text:
            raise ValueError("Text is empty, please try again")
        #else: return self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)

        else:
            text_list = []
            text_list.append(text)
            print(text_list)
            model_result = self.model.encode(text_list)
            #embedding = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            #print("#-0.035")
            return model_result[0]

    def build_embeddings(self, documents:list):
        self.documents = documents
        document_list = []
        for document in documents:
        # this is working with the given movie-dictionaries
            self.document_map[document["id"]] = document
            document_string = f"{document['title']}: {document['description']}"
            document_list.append(document_string)
        self.embeddings = self.model.encode(document_list, show_progress_bar=True, batch_size=64)
        #self.embeddings = self.model.encode(document_list,show_progress_bar=True,batch_size=64,normalize_embeddings=True,convert_to_numpy=True)
        
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

    def search(self, query, limit):
        if self.embeddings is None or np.size(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
            print(ValueError)
            return None
                    
        else:
            # print(self.embeddings.shape) 
            # print(len(self.documents))
            embedded_query = self.generate_embedding(query)
            full_movie_list = []
            
            for document, embedding in zip(self.documents, self.embeddings):
                movie = {}
                similarity_score = cosine_similarity(embedded_query, embedding)
                #similarity_score_list.append((similarity_score, document))
                movie["score"] = similarity_score
                movie["title"] = document["title"]
                movie["description"] = document["description"]
                full_movie_list.append(movie)
    
        
            #same as:
            #def get_score(item):
            #    return item["score"]
            #full_movie_list.sort(key=get_score)

            full_movie_list.sort(key=lambda item: item["score"], reverse=True)
            
            return full_movie_list[0:limit]
            
        

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