

#always source .venv/bin/activate
import json
import string
from search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords, project_root, create_subfolder
from nltk.stem import PorterStemmer
import pickle
from collections import Counter



def test_text(text: str) -> str:
    return clean(text)

def clean(query: str) -> str:
    #remove_punctuation = str.maketrans("", "", string.punctuation)
    remove_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    # make translation table, delete all punctuations:
    # string.punctuation = !"#$%&'()*+,-./:;<=>?@[]^_`{|}~\
    # make translation table, e.g. str.maketrans(from, to, delete)
    # str.maketrans("aeiou", "AEIOU", "!?.")
    cleaned_query = query.translate(remove_punctuation).lower().strip()
    
    return cleaned_query

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
        self.index = {} #a dictionary mapping tokens (strings) to sets of document IDs (integers)
        self.docmap = {} #a dictionary mapping document IDs to their full document objects.
        self.term_frequencies = {}
        self.index_path = project_root / "cache" / "index.pkl"
        self.docmap_path = project_root / "cache" / "docmap.pkl"
        self.term_frequencies_path = project_root / "cache" / "term_frequencies.pkl"
        print("Objekt succesfully created")
    
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
                
    def add_document(self, doc_id, text): #TODO
        text = text_pipeline(text)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        
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
        

    def get_document(self, term): #TODO
        
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
        except IOError as e:
            print(f"Error saving file: {e}")

    def get_tf(self, doc_id, term):
        return self.term_frequencies[doc_id][term]
        
#CALLS: only once by executing search.py
stopwords = load_stopwords()
stemmer = PorterStemmer()