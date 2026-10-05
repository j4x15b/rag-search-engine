import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies, DEFAULT_TEXT_LENGTH

######################################################################################
# CLI FUNCTIONS
#
######################################################################################

def normalize(list_of_scores: list):
    #takes an input string of numbers, normalizes them with min/max, returns a normalized list
    try:
        min_score = min(list_of_scores)
        max_score = max(list_of_scores)
        #print(f"Liste mit Werten: {list_of_scores}")
        if min_score == max_score:
            normalized_list = [1.0 for _ in list_of_scores]
        else:
            normalized_list = [((score - min_score) / (max_score - min_score)) for score in list_of_scores]
        
        #print(f"normalisierte Liste: {normalized_list}")

        return normalized_list

    except ValueError:
        print("Ungültige Eingabe")
    # list_of_scores = []
    # if scores:
    #     list_of_scores = [float(x.strip()) for x in scores.split(",")]
        # for score in scores:
        #     if int(score):
        #         list_of_scores.append(int(score))
        
def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    # takes a bm25-score and a semantic-score and returns a hybrid-score
    return alpha * bm25_score + (1 - alpha) * semantic_score

    """
    α = 1.0: [████████████████████] 100% Keyword
    α = 0.7: [██████████████------] 70% Keyword, 30% Semantic
    α = 0.5: [██████████----------] 50/50 Split
    α = 0.2: [████----------------] 20% Keyword, 80% Semantic
    α = 0.0: [--------------------] 100% Semantic
    
    alpha (or "α") is just a constant that we can use to dynamically control the weighting between the two scores:
    
    Query Type	    Example	Chosen  Alpha	Reason
    Exact match	    "The Revenant"	0.8	    Title search needs keywords
    Conceptual	    "family movies"	0.2	    Meaning matters more
    Mixed	        "2015 comedies"	0.5	    Both year AND concept

    """

def rrf_score(rank: int, k: int = 60) -> float:
    """
    Brother Bear – 1 / (60 + 1) = 0.0164
    Jungle Book – 1 / (60 + 2) = 0.0161
    Paddington – 1 / (60 + 3) = 0.0159
    Semantic:
    Paddington – 1 / (60 + 1) = 0.0164
    Brother Bear – 1 / (60 + 2) = 0.0161
    We Bare Bears – 1 / (60 + 3) = 0.0159
    """        
    return 1 / (k + rank)

def weighted_search_command(query, alpha=0.5, limit=5):
    movie_list = load_movies()
    working_limit = min(limit*500, len(movie_list))
    hybrid_search = HybridSearch(movie_list)
    
    results = hybrid_search.weighted_search(query, alpha, limit)
    for i, result in enumerate(results[:limit]):
            print(f"\n{i+1}. {result['title']}")
            print(f" BM25: {result['normalized_bm25_score']:.3f}, Semantic: {result['normalized_semantic_score']:.3f}")
            print(f" {result['description']}")

#######################################################################################
# CLASSES
#
#######################################################################################
class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load() #loads InvertedIndex
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        keyword_results = self._bm25_search(query, limit)
        keyword_id_order = [x[0] for x in keyword_results] 
        keyword_scores = [x[1] for x in keyword_results]
        #print(keyword_results)
        semantic_results = self.semantic_search.search_chunks(query, limit)
        semantic_id_order = [x['id'] for x in semantic_results]
        semantic_scores = [x['score'] for x in semantic_results]
        #print(semantic_results)
        normalized_keyword_scores = normalize(keyword_scores)
        normalized_semantic_scores = normalize(semantic_scores)

        #print(normalized_keyword_scores)
        #print(normalized_semantic_scores)
        scores_dict = {}
        #print(self.documents[1]['title'])

        #building Dict for all keyword results
        for id, result, normalized_result in zip(keyword_id_order, keyword_scores, normalized_keyword_scores):
            scores_dict[id] = { 
                'bm25_score' : result, 
                'normalized_bm25_score' : normalized_result, 
                'title' : self.documents[id-1]['title'],
                'semantic_score' : 0.0,
                'normalized_semantic_score' : 0.0,
                'hybrid_score' : 0.0,
                'description' : self.documents[id-1]['description'][:DEFAULT_TEXT_LENGTH],
            }

        #updating Dict with semantic scores:
        #print(scores_dict[2068])
        #print(scores_dict[1649])

        #updating dict-entries for existing ids, building new dict-entries for new ids
        for id, result, normalized_result in zip(semantic_id_order, semantic_scores, normalized_semantic_scores):
            if id in scores_dict:
                scores_dict[id]['semantic_score'] = result
                scores_dict[id]['normalized_semantic_score'] = normalized_result

            else:
                scores_dict[id] = { 
                    'bm25_score' : 0.0, 
                    'normalized_bm25_score' : 0.0, 
                    'title' : self.documents[id-1]['title'],
                    'semantic_score' : result,
                    'normalized_semantic_score' : normalized_result,
                    'hybrid_score' : 0.0,
                    'description' : self.documents[id-1]['description'][:DEFAULT_TEXT_LENGTH],
            }


        for id in scores_dict:
            #print(scores_dict[id]['normalized_bm25_score'], scores_dict[id]['normalized_semantic_score'])
            scores_dict[id]['hybrid_score'] = hybrid_score(scores_dict[id]['normalized_bm25_score'], scores_dict[id]['normalized_semantic_score'])

        # import itertools
        # print(dict(itertools.islice(sorted_scores_dict.items(), 10)))
        
        sorted_scores_dict = sorted(scores_dict.values(), key=lambda item: item['hybrid_score'], reverse=True)
        
        return sorted_scores_dict

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        
    