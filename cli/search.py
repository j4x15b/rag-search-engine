#next up: tokenization

#always source .venv/bin/activate
import json
import string
from search_utils import DEFAULT_SEARCH_LIMIT, load_movies

def tokenize(query: str) -> list[str]:
    # input string, split, clean and put in list
    # .split() on whitespace
    # Remove any empty tokens
    pass

def search_command(search_word: str, list_limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    search_result = []
    
    remove_punctuation = str.maketrans("", "", string.punctuation)
    
    # make translation table, delete all punctuations:
    # string.punctuation = !"#$%&'()*+,-./:;<=>?@[]^_`{|}~\
    # make translation table, e.g. str.maketrans(from, to, delete)
    # str.maketrans("aeiou", "AEIOU", "!?.")

    cleaned_search_word = search_word.translate(remove_punctuation).lower()
    


    movie_data = load_movies()
    # search movie titles: cleaned search word and cleaned movie title
    for movie in movie_data:
        cleaned_title = movie["title"].translate(remove_punctuation).lower()
        if cleaned_search_word in cleaned_title:
            search_result.append(movie)
            if len(search_result) >= list_limit:
                break

    return search_result