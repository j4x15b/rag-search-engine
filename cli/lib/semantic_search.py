#if I need better results: late chunking or ColBERT


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#no NVIDIA CUDA Devices, so it works with CPU
#os.environ["TRANSFORMERS_OFFLINE"] = "1"
#os.environ["HF_DATASETS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from .search_utils import load_movies, cache_path, create_subfolder, format_search_result

import numpy as np

# Load the model (downloads automatically the first time)

offline_mode = 1 #os.getenv("OFFLINE_MODE") == "1"

#####################################################################
## CLI Methods
#####################################################################

def search_command(query, limit=5):
    #print("Hellooo")
    #print("LIMIT:", limit)
    print(limit)
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
    #print("SEMANTIC CHUNKING")
    chunk_list = []
    text = text.strip()
    if not text:
        print("Text is empty")
        return chunk_list
    text_list = re.split(r"(?<=[.!?])\s+",text)
    last_chunk = text_list[-1].strip()
    print(last_chunk)
    #if the text_list contains more than one chunk AND the last chunk has no punctuation mark: put the last and the second-last together
    if len(text_list) > 1 and not re.search(r"[.!?]$", last_chunk):
        print("Der String endet nicht mit einem Satzzeichen. Letzte Chunks zusammengefasst")
        second_last_chunk = text_list[-2].strip()
        text_list[-2] = f"{second_last_chunk} {last_chunk}"
        text_list.pop()

    #print(text_list)
    chunk_size = max_chunk_size
    chunk_string = ""
    
    if overlap >= max_chunk_size:
        overlap = max_chunk_size - 1
        print(f"Overlap has been overwritten to {overlap}")
        
    #statistical chunking: measure text length and sentence length to produce comparable chunks
    #sentences = re.split(r"(?<=[.!?])\s+", text)
    #avg_len = sum(len(s) for s in sentences) / len(sentences)
    #total_len = len(text)

    #semantical chunking: 
    #Split into sentences.
    #Embed each sentence into a vector.
    #Walk through consecutive sentences and measure cosine similarity between neighbors.
    #When similarity drops below a threshold, that's a topic shift, so cut a new chunk there.
    
    #split at meaning-boundaries 

    # e.g. aim for chunks that hold a few "average" sentences
    #target_len = avg_len * 4

    # #schleife
    # start = 0
    # end = start + chunk_size
    # 
    # while version of the loop (outdated)
    # while start <= len(text_list):
    #     chunk = " ".join(text_list[start:end])
    #     if chunk: chunk_list.append(chunk)
    #     if end >= len(text_list): break
    #     start = end - overlap #ende ist exklusiv, deshalb ist ende das nächste wort - minus overlap
    #     end = start + chunk_size        
    
    #calculating the amount of chunks:
    import math
    step = chunk_size - overlap
    num_chunks = math.ceil((len(text_list) - overlap) / step)

    #schleife
    start = 0
    end = start + chunk_size
    for i in range(num_chunks):
        #print(f"start: {start}, end: {end}")
        chunk = " ".join(text_list[start:end])
        if chunk: chunk_list.append(chunk)

        #calculating new start and end:
        start = start + step
        end = start + chunk_size

    return chunk_list

def embed_chunks_command():
    # semantic_search = SemanticSearch()
    chunked_semantic_search = ChunkedSemanticSearch()
    movie_list = load_movies()
    chunked_semantic_search.load_or_create_chunk_embeddings(movie_list)
    print(f"Generated {len(chunked_semantic_search.chunk_embeddings)} chunked embeddings")

def build_chunk_embeddings():
    # semantic_search = SemanticSearch()
    chunked_semantic_search = ChunkedSemanticSearch()
    movie_list = load_movies()
    chunked_semantic_search.build_chunk_embeddings(movie_list)
    #print(f"Generated {len(chunked_semantic_search.chunk_embeddings)} chunked embeddings")
    
    
    """
    results = semantic_search.search(query, limit)
    for i, result in enumerate(results):
        #print(f"{i+1}. {result["title"]} (score: {result["score"]:.4f}) \n {result["description"]}\n\n")
        print(f"{i+1}. {result["title"]} (score: {result["score"]:.4f}) \n")
    """

def search_chunked_command(query: str, limit=5):
    chunked_semantic_search = ChunkedSemanticSearch()
    movie_list = load_movies()
    chunked_semantic_search.load_or_create_chunk_embeddings(movie_list)
    results = chunked_semantic_search.search_chunks(query, limit)
    #print(results)
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")
#####################################################################
## Working Methods
#####################################################################
def embed_text(text):
    semantic_search = SemanticSearch()
    embedded_text = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedded_text[:3]}")
    print(f"Dimensions: {embedded_text.shape[0]}")

    return embedded_text

def embed_query(query):
    semantic_search = SemanticSearch()
    embedded_query = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedded_query[:3]}")
    print(f"Shape: {embedded_query.shape}")
    
    return embedded_query

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

def verify_chunk_embeddings():
    chunked_semantic_search = ChunkedSemanticSearch()
    movie_list = load_movies()
    chunked_semantic_search.load_or_create_chunk_embeddings(movie_list)

    print(f"Number of docs:   {len(movie_list)}")
    print(f"Embeddings shape: {chunked_semantic_search.chunk_embeddings.shape[0]} vectors in {chunked_semantic_search.chunk_embeddings.shape[1]} dimensions")


#####################################################################
## Working Classes (haha)
#####################################################################

class SemanticSearch():
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        #if self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.model = SentenceTransformer(model_name, local_files_only=offline_mode)#, device="cpu"))
        #self.model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=offline_mode)#, device="cpu")
        
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

#sublasses 

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        #super().__init__(model_name)
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = cache_path / "chunk_embeddings.npy"
        self.chunk_metadata_path = cache_path / "chunk_metadata.json"

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        print("building chunk embeddings")
        if self.chunk_metadata_path.exists(): 
            os.remove(self.chunk_metadata_path)
        if self.chunk_embeddings_path.exists(): 
            os.remove(self.chunk_embeddings_path)
        self.documents = documents
        chunk_list = [] #string
        list_of_all_chunks = [] #string
        chunk_metadata_list = [] #dict
        i = 1
        for document in documents:
        # this is working with the given movie-dictionaries
            if not document['description']: pass
            else:
                self.document_map[document["id"]] = document
                print(document['id'])
                #print('\n')
                #print(document['description'])
                
                chunk_list = semantic_chunk_command(document['description'], 4, 1)
                for i, chunk in enumerate(chunk_list):
                    list_of_all_chunks.append(chunk)
                    chunk_metadata_list.append(
                        {
                            "movie_idx":document["id"], #movie_id
                            "chunk_idx":i, #actual chunk number
                            "total_chunks":len(chunk_list), #number of chunks in the movie
                        })
                #print(chunk_list[0])
                #print(chunk_list)
                i += 1
            
        # embed the generated chunk list 
        print("encoding:")
        self.chunk_embeddings = self.model.encode(list_of_all_chunks, show_progress_bar = True)
        # save chunk_metadata_list as class object
        self.chunk_metadata = chunk_metadata_list
        print(len(self.chunk_embeddings))
        #save embedded chunk list and metadata
        print("Saving...")
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        import json
        with open(self.chunk_metadata_path, "w") as f:
            json.dump(self.chunk_metadata, f, indent=2)
            #json.dump({"chunks": self.chunk_metadata, "total_chunks": len(list_of_all_chunks)}, f, indent=2)
        
        
        return self.chunk_embeddings
        
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        #loads the chunked embeddings, if exist - else builds them
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        
        if self.chunk_embeddings_path.exists():
            print("loading chunk embeddings")
            self.chunk_embeddings = np.load(self.chunk_embeddings_path, self.chunk_embeddings)
            #print(self.chunk_embeddings)
            import json
            with open(self.chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings

        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        print(limit)
        embedded_query = self.generate_embedding(query)
        chunk_score = []
        if 0 == 1: raise ValueError("ERROR, ERROR, ERROR")
        #if not self.chunk_embeddings: raise Error("No Chunks embedded, please run command load_or_create_chunks")
        else:
            for i, (chunk, chunk_metadata) in enumerate(zip(self.chunk_embeddings, self.chunk_metadata)):
                #print(i)
                chunk_cosine_score = cosine_similarity(embedded_query, chunk)
                chunk_score_dict = {
                    "chunk_idx" : i, #actual chunk number
                    "movie_idx" : chunk_metadata["movie_idx"], #movie_id
                    "score" : float(chunk_cosine_score), #number of chunks in the movie
                }
                chunk_score.append(chunk_score_dict)
                #if i > 200: break

        #print(chunk_score)
        movie_idx_score_dict = {}
        
        # looking for max score
        # Sum of Scores (with #chunks normalization), Average Score or Max Score - I use Max Score here
        for score in chunk_score:
            if score["movie_idx"] not in movie_idx_score_dict:
                movie_idx_score_dict[score["movie_idx"]] = score["score"]
            if score["score"] > movie_idx_score_dict[score["movie_idx"]]:
                movie_idx_score_dict[score["movie_idx"]] = score["score"]
        
        #print(movie_idx_score_dict)

        #sorting by score        
        sorted_movie_idx_score_dict = {k: v for k, v in sorted(movie_idx_score_dict.items(), reverse=True, key=lambda item: item[1])}
        #print(sorted_movie_idx_score_dict)

        result = dict(list(sorted_movie_idx_score_dict.items())[:limit])
        print(result)
        formatted_result = format_search_result(result, self.document_map)
        #full_result_dict = {}
        #print(formatted_result)
        return formatted_result

        """ 
        {
            "id": doc_id,
            "title": title,
            "document": document[:100],
            "score": round(score, SCORE_PRECISION),
            "metadata": metadata or {},
        }
        
        print(short_list)
        return result_list

        """