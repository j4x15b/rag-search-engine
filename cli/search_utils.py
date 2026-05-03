import json
from pathlib import Path
from functools import lru_cache

# output movie count
DEFAULT_SEARCH_LIMIT = 5

#setting data_paths
project_root = Path(__file__).parent.parent
movie_data_path = project_root / "data" / "movies.json"
stopwords_data_path = project_root / "data" / "stopwords.txt"

#creating subpath

def create_subfolder(folder_name: str) -> bool:
    target = project_root / folder_name
    if target.exists():
        print("Folder already existing")
        return True
    else:
        print(f"creating folder: {folder_name}")
        target.mkdir()
        return False

# load  movie data base
@lru_cache(maxsize=None)
def load_movies() -> list[dict]:
    with open(movie_data_path, 'r', encoding='utf-8') as file:
        movie_data = json.load(file)
    return movie_data["movies"]

@lru_cache(maxsize=None)
def load_stopwords(): # -> set(str)
    with open(stopwords_data_path, 'r', encoding='utf-8') as file:
        stopword_set = frozenset(line.strip() for line in file)
    return stopword_set

