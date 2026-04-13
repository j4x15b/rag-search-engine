#next up: tokenization

#always source .venv/bin/activate
import json
import string
from search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords
from nltk.stem import PorterStemmer

def clean(query: str) -> str:
    remove_punctuation = str.maketrans("", "", string.punctuation)
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
    
    # Reihenfolge:
    #CLEAN, TOKENIZE, FILTER STOPWORDS, STEM

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
        #print(stemmed_filtered_tokenized_title)
        for token in stemmed_tokenized_query:
            if token in stemmed_filtered_tokenized_title:
                search_result.append(movie)
                break
        if len(search_result) >= list_limit:
            break

    return search_result