import json
from pathlib import Path
from functools import lru_cache
import string

# output movie count
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_TEXT_LENGTH = 200
FULL_TEXT_LENGTH = None


# BM25_TFIDF constants
BM25_K1 = 1.5
BM25_B = 0.75

#setting data_paths
project_root = Path(__file__).resolve().parent.parent
cache_path = project_root / "cache"

HF_CACHE_REPO_ID = "j4x15b/movie-search-cache"
HF_CACHE_FILES = (
    "chunk_embeddings.npy",
    "chunk_metadata.json",
    "doc_lengths.pkl",
    "docmap.pkl",
    "index.pkl",
    "movie_embeddings.npy",
    "term_frequencies.pkl",
)

def ensure_hf_cache(token: str | None = None) -> None:
    """Ensure the prebuilt search cache exists locally, downloading missing files from HF."""
    from huggingface_hub import hf_hub_download

    cache_path.mkdir(parents=True, exist_ok=True)
    missing = [name for name in HF_CACHE_FILES if not (cache_path / name).is_file()]
    if not missing:
        return
    if not token:
        raise RuntimeError(
            "Search cache files are missing and the Hugging Face dataset is private. "
            "Add HF_TOKEN to Streamlit Secrets, or provide the cache files locally. "
            f"Missing: {', '.join(missing)}"
        )

    for filename in missing:
        hf_hub_download(
            repo_id=HF_CACHE_REPO_ID,
            filename=filename,
            repo_type="dataset",
            token=token,
            local_dir=str(cache_path),
        )

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
        target.mkdir(parents=True)
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
        #stopword_set = frozenset(line.strip() for line in file)
        stopword_set = frozenset(line.strip().lower().translate(str.maketrans("", "", string.punctuation)) for line in file)
        #returns lowered, stripped stopwords with removed punctuation
    return stopword_set

# formatting search results

def format_search_result(result, document_map):
    #formatted_result = format_search_result(result, self.document_map, self.chunk_metadata)    
    
    result_list = []
    #print(self.chunk_metadata)
    for key, value in result.items():
        result_list.append(
        #full_result_dict[key] = {
        {
            "id": key,
            "title": document_map[key]["title"],
            "description": document_map[key]["description"],
            #"document": document_map[key]["description"],
            "score": round(value, 4) #"score": round(value, SCORE_PRECISION),
            #"metadata":  ???
        })
 
    return result_list
