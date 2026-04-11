#next up: tokenization

#always source .venv/bin/activate
import json
import string
from search_utils import DEFAULT_SEARCH_LIMIT, load_movies

def clean_query(query: str) -> str:
    remove_punctuation = str.maketrans("", "", string.punctuation)
    # make translation table, delete all punctuations:
    # string.punctuation = !"#$%&'()*+,-./:;<=>?@[]^_`{|}~\
    # make translation table, e.g. str.maketrans(from, to, delete)
    # str.maketrans("aeiou", "AEIOU", "!?.")
    cleaned_query = query.translate(remove_punctuation).lower()
    
    return cleaned_query

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

def search_command(search_word: str, list_limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    search_result = []
    
    cleaned_query = clean_query(search_word)
    tokenized_query = tokenize(cleaned_query)

    movie_data = load_movies()
    # search movie titles: cleaned search word and cleaned movie title
    for movie in movie_data:
        cleaned_title = clean_query(movie["title"])
        for token in tokenized_query:
            if token in cleaned_title:
                search_result.append(movie)
                break
        if len(search_result) >= list_limit:
            break

    return search_result