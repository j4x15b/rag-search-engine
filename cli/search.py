#next up: tokenization

#always source .venv/bin/activate
import json
import string
from search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords, project_root, create_subfolder
from nltk.stem import PorterStemmer
import pickle

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
    stopwords = load_stopwords()
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
    stemmer = PorterStemmer()
    for word in query:
        stemmed_list.append(stemmer.stem(word))

    return stemmed_list

def search_command(query: str, list_limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    search_result = []
    
    # order: CLEAN, TOKENIZE, FILTER STOPWORDS, STEM

    # clean from punctuation and space
    cleaned_query = clean(query)
    # tokenize
    tokenized_query = tokenize(cleaned_query)
    # removing_stopwords
    filtered_tokenized_query = remove_stopwords(tokenized_query)
    # stemming
    stemmed_tokenized_query = stem(filtered_tokenized_query)
    movie_data = load_movies()
    # search movie titles: cleaned search word and cleaned movie title
    for movie in movie_data:
        cleaned_title = clean(movie["title"])
        tokenized_title = tokenize(cleaned_title)
        filtered_tokenized_title = remove_stopwords(tokenized_title)
        stemmed_filtered_tokenized_title = stem(filtered_tokenized_title)
        
        for token in stemmed_tokenized_query:
            if token in stemmed_filtered_tokenized_title:
                search_result.append(movie)
                break
        if len(search_result) >= list_limit:
            break

    return search_result


class InvertedIndex():
    def __init__(self):
        self.index = {} #a dictionary mapping tokens (strings) to sets of document IDs (integers)
        self.docmap = {} #a dictionary mapping document IDs to their full document objects.
        print("Objekt succesfully created")
        self.index['Test1'] = 1
        self.index['Test2'] = 2
        self.docmap["Hello"] = 'Test'
        self.docmap["Test"] = 'Test_One_Two'

    def __add_document(self, doc_id, text):
        pass
        print("__add_document method")
        # Tokenize the input text, then add each token to the index with the document ID.
        cleaned_text = clean(text)
        tokenized_text = tokenize(cleaned_text)
        for token in tokenized_text:
            #print(token)
            self.index[token] = doc_id
        print("end_add_document_method")
    def get_document(self, doc_id, text):
        pass
        #It should get the set of document IDs for a given token, and return them as a list, 
        # sorted in ascending order. For our purposes, you can assume that the input term is a 
        # single word/token – though you may still want to lowercase it for good measure.
    def build(self):
        print("building inverted index method")
        movie_data = load_movies()
        for movie in movie_data:
            id = movie['id']
            movie_text = f"{movie['title']} {movie['description']}"
            self.__add_document(id, movie_text)
        # It should iterate over all the movies and add them to both the index and the docmap.
        # When adding the movie data to the index with __add_document(), concatenate the title and the 
        # description and use that as the input text. For example:
        # f"{m['title']} {m['description']}"
    def save(self):
        #  should save the index and docmap attributes to disk using the pickle module's dump function.
        # Use the file path/name cache/index.pkl for the index.
        # Use the file path/name cache/docmap.pkl for the docmap.
        # Have this method create the cache directory if it doesn't exist (before trying to write files into it).
        print("saving file method")
        
        if create_subfolder("cache"):
            print("Folder already exists")
        else:
            print("Cache folder created")
        index_path = project_root / "cache" / "index.pkl"
        docmap_path = project_root / "cache" / "docmap.pkl"
        if index_path.exists() or docmap_path.exists():
            loop = True
            while loop:
                overwrite_choice = input("Overwrite? (y/n): ").lower()
                if overwrite_choice == "y":
                    loop = False
                    try:
                        with open(index_path, 'wb') as file:
                            pickle.dump(self.index, file)
                        with open(docmap_path, 'wb') as file:
                            pickle.dump(self.docmap, file)
                    except IOError as e:
                        print(f"Error saving file: {e}")

                elif overwrite_choice == "n":
                    loop = False
                    print("Saving proces aborted")
        else:
            try:
                with open(index_path, 'wb') as file:
                    pickle.dump(self.index, file)
                with open(docmap_path, 'wb') as file:
                    pickle.dump(self.docmap, file)
            except IOError as e:
                print(f"Error saving file: {e}")

        with open(index_path, 'rb') as file:
            a = pickle.load(file)
        print(a)


        
