
#always source .venv/bin/activate


# import libraries
import json
import string
import math

from nltk.stem import PorterStemmer
import pickle
from collections import Counter
from collections import defaultdict

# import helper functions
from search_utils import load_movies, load_stopwords, project_root, create_subfolder
#import constants
from search_utils import DEFAULT_SEARCH_LIMIT, BM25_K1, BM25_B




def test_text(text: str) -> str:
    return clean(text)

def clean(query: str) -> str:
    remove_punctuation = str.maketrans("", "", string.punctuation)
    # removes punctuation entirely

    #remove_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    # removes punctuation but also splits words 
    # make translation table, delete all punctuations:
    # string.punctuation = !"#$%&'()*+,-./:;<=>?@[]^_`{|}~\
    # make translation table, e.g. str.maketrans(from, to, delete)
    # str.maketrans("aeiou", "AEIOU", "!?.")
    cleaned_query = query.translate(remove_punctuation).lower().strip()
    
    return cleaned_query

def print_document(doc_id):
    # prints the whole content of the document with the given document id
    inverted_index = InvertedIndex()
    inverted_index.load()
    print(f"Length: {inverted_index.doc_lengths[doc_id]}")
    print(inverted_index.docmap[doc_id])
    

def remove_stopwords(query_tokens: list[str]) -> list[str]:
    filtered_tokens = []
    for token in query_tokens:
        if token not in stopwords:
            filtered_tokens.append(token)
    return filtered_tokens

def tokenize(query: str) -> list[str]:
    # input string, split, clean and put in list
    # .split() on whitespace
    # Remove any empty tokens
    tokenized_query = []
    temp_list = query.split()
    for item in temp_list:
        if item:
            tokenized_query.append(item)

    return tokenized_query

def stem(query: list[str]) -> list[str]:
    stemmed_list = []
    for word in query:
        stemmed_list.append(stemmer.stem(word))

    return stemmed_list

def text_pipeline(text: str) -> list[str]:
    # order: CLEAN, TOKENIZE, FILTER STOPWORDS, STEM
    text = clean(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)

    return tokens

def single_term_tokenizer(text: str) -> list[str]:
    if len(text.split()) > 1:
        raise Exception("Only one word allowed")
    else:
        tokenized_term_list = text_pipeline(text)
        term = " ".join(tokenized_term_list)
        return term

def bm25_idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    tokenized_term = single_term_tokenizer(term)
    bm25_idf = inverted_index.get_bm25_idf(tokenized_term)
    return bm25_idf

def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    tokenized_term = single_term_tokenizer(term)
    bm25_tf = inverted_index.get_bm25_tf(doc_id, tokenized_term, k1, b)

    return bm25_tf
    
def search_command(inverted_index, query: str, list_limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    
    query = text_pipeline(query)
    # movie_list = load_movies()
    # movie_id_index = {movie["id"]: movie for movie in movie_list}

    print(query)
    result_set = set()
    search_result = []

    for token in query:
        print(token)
        for doc_id in inverted_index.get_document(token):
            if doc_id in result_set:
                continue
            result_set.add(doc_id)
            search_result.append(inverted_index.docmap[doc_id]["title"])
            if len(search_result) >= list_limit:
                return search_result
    return search_result

class InvertedIndex():
    def __init__(self): #OK
        #self.index = {} #a dictionary mapping tokens (strings) to sets of document IDs (integers)
        self.index = defaultdict(set)
        self.docmap = {} #a dictionary mapping document IDs to their full document objects.
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.index_path = project_root / "cache" / "index.pkl"
        self.docmap_path = project_root / "cache" / "docmap.pkl"
        self.term_frequencies_path = project_root / "cache" / "term_frequencies.pkl"
        self.doc_lengths_path = project_root / "cache" / "doc_lengths.pkl"
        print("debug: InvertedIndex-object succesfully created")
    
    def load(self):        
        #load index-file
        with open(self.index_path, 'rb') as file:
            f = pickle.load(file)
            self.index = f
        #load docmap-file
        with open(self.docmap_path, 'rb') as file:
            f = pickle.load(file)
            self.docmap = f
        with open(self.term_frequencies_path, 'rb') as file:
            f = pickle.load(file)
            self.term_frequencies = f
        with open(self.doc_lengths_path, 'rb') as file:
            f = pickle.load(file)
            self.doc_lengths = f
        print("index, docmap, term frequencies & doc lengths loaded")
                
    def add_document(self, doc_id, text):
        #add document to docID
        text = text_pipeline(text) #tokenize text
        total_tokens = 0
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        if doc_id not in self.doc_lengths:
            self.doc_lengths[doc_id] = 0
        self.doc_lengths[doc_id] = len(text)
    
        for token in text:
            self.term_frequencies[doc_id][token] += 1
            #self.index.setdefault(token, set()).add(doc_id)
            #same as:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}

        #print("end_add_document_method")
        #print(self.index)
        

    def get_document(self, term):
        #returns document IDs of search term from movie-list
        sorted_movie_id_list = []
        if term in self.index:
            return sorted(list(self.index[term]))
        else: 
            return []
        #It should get the set of document IDs for a given token, and return them as a list, 
        # sorted in ascending order. For our purposes, you can assume that the input term is a 
        # single word/token – though you may still want to lowercase it for good measure.

    def build(self): 
        #OK
        print("Building index and Docmap")
        movie_data = load_movies()
        # movie_data is a list of dicts{'id', 'title', 'description'}
        for movie in movie_data:
            doc_id = movie['id']
            movie_text = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.add_document(doc_id, movie_text)
        # It should iterate over all the movies and add them to both the index and the docmap.
        # When adding the movie data to the index with __add_document(), concatenate the title and the 
        # description and use that as the input text. For example:
        # f"{m['title']} {m['description']}"
    
    def save(self): #OK
        #  should save the index and docmap attributes to disk using the pickle module's dump function.
        # Use the file path/name cache/index.pkl for the index.
        # Use the file path/name cache/docmap.pkl for the docmap.
        # Have this method create the cache directory if it doesn't exist (before trying to write files into it).
        print("saving file method")

        # if create_subfolder("cache"):
        #     print("Folder already exists")
        # else:
        #     print("Cache folder created")
        
        #if self.index_path.exists() or self.docmap_path.exists():
            # loop = True
        #     while loop:
        #         overwrite_choice = input("Overwrite? (y/n): ").lower()
        #         if overwrite_choice == "y":
        #             loop = False
        #             try:
        #                 with open(index_path, 'wb') as file:
        #                     pickle.dump(self.index, file)
        #                 with open(docmap_path, 'wb') as file:
        #                     pickle.dump(self.docmap, file)
        #             except IOError as e:
        #                 print(f"Error saving file: {e}")

        #         elif overwrite_choice == "n":
        #             loop = False
        #             print("Saving process aborted")
        # else:
        try:
            with open(self.index_path, 'wb') as file:
                pickle.dump(self.index, file)
            with open(self.docmap_path, 'wb') as file:
                pickle.dump(self.docmap, file)
            with open(self.term_frequencies_path, 'wb') as file:
                pickle.dump(self.term_frequencies, file)
            with open(self.doc_lengths_path, 'wb') as file:
                pickle.dump(self.doc_lengths, file)
                
        except IOError as e:
            print(f"Error saving file: {e}")

    def get_tf(self, doc_id, term):
        return self.term_frequencies[doc_id][term]
    
    def get_bm25_idf(self, term:str) -> float:
        #calculates the bm25_idf, improved and smoothed tf_idf rate
        N = len(self.docmap)
        df = len(self.index[term])
        bm25_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        # N - df + 0.5: Count of documents WITHOUT the term
        # / df: Count of documents WITH the term
        # +0.5: smoothing, so you don't have division by 0
        # log: smoothes the curve for large numbers
        
        return bm25_idf

    def get_bm25_tf(self, doc_id:int, term:str, k1=BM25_K1, b = BM25_B) -> float:
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        
        tf = self.get_tf(doc_id, term)
        #bm25 formula for tf-saturation
        #bm25_tf = (tf * (k1 + 1)) / (tf + k1)
        bm25_tf_norm = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return bm25_tf_norm
    
    def __get_avg_doc_length(self) -> float:
        # calculates and returns the average document-length
        N = len(self.doc_lengths) #N = total document count        
        avg_doc_length = None
        if N < 1: return 0.0
        else:
            total_doc_length = sum(self.doc_lengths.values())
            avg_doc_length = total_doc_length / N
            
        return avg_doc_length
    
    def bm25(self, doc_id, term) -> float:
        # calculates and returns the bm25 score of a given document and term-pair
        bm25 = self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)
        return bm25
    
    def bm25_search(self, query:str, limit=5):
        # performs bm25_search and returns the top results
        tokenized_query = text_pipeline(query)
        score_dict = {} #score dictionary with document IDs:bm25-score
        
        for token in tokenized_query:
            for doc_id in self.docmap:            
                bm25_score = self.bm25(doc_id, token)
                if doc_id == 2275:
                    print(f"token={token}, doc_id={doc_id}, score={bm25_score}")
                #print(token, doc_id, bm25_score)            
                if doc_id not in score_dict:
                    score_dict[doc_id] = 0
                score_dict[doc_id] += bm25_score
        print(score_dict[2275])
        sorted_dict = {k: v for k, v in sorted(score_dict.items(), reverse=True, key=lambda item: item[1])}
        #print(sorted_dict)
        items = list(sorted_dict.items())
        
        return items[:limit]
        
        
#CALLS: only once by executing search.py
stopwords = load_stopwords()
stemmer = PorterStemmer()