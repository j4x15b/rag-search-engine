import json
from pathlib import Path

# output movie count
DEFAULT_SEARCH_LIMIT = 5

#setting data_paths
project_root = Path(__file__).parent.parent
movie_data_path = project_root / "data" / "movies.json"

# load  movie data base
def load_movies() -> list[dict]:
        with open(movie_data_path, 'r', encoding='utf-8') as file:
            movie_data = json.load(file)
        return movie_data["movies"]

