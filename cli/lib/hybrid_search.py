import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies

######################################################################################
# CLI FUNCTIONS
#
######################################################################################

def normalize(list_of_scores: list):
    #takes an input string of numbers, normalizes them with min/max, returns a normalized list
    try:
        min_score = min(list_of_scores)
        max_score = max(list_of_scores)
        print(f"Liste mit Werten: {list_of_scores}")
        if min_score == max_score:
            normalized_list = [(score/score) for score in list_of_scores]
        else:
            normalized_list = [((score - min_score) / (max_score - min_score)) for score in list_of_scores]
        
        print(f"normalisierte Liste: {normalized_list}")

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

def weighted_search_command(query, alpha=0.5, limit=5):
    movie_list = load_movies()
    hybrid_search = HybridSearch(movie_list)
    
    # chunked_semantic_search = ChunkedSemanticSearch()
    
    # chunked_semantic_search.load_or_create_chunk_embeddings(movie_list)
    # results = chunked_semantic_search.search_chunks(query, limit)
    
    results = hybrid_search.weighted_search(query, alpha, limit)

    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']}")
        print(f"Hybrid Score: ")
        print(f"BM25: , Semantic: ")
        print(f"{result['description']}")
"""
        1. Paddington
    Hybrid Score: 1.000
    BM25: 1.000, Semantic: 1.000
    Deep in the rainforests of Peru, a young bear lives peacefully with his Aunt Lucy and Uncle Pastuzo,...
    2. The Indian in the Cupboard
    Hybrid Score: 0.943
    BM25: 0.966, Semantic: 0.850
    On his ninth birthday, Omri receives an old cupboard from his brother Gillon (Vincent Kartheiser) an...
"""

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
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        keyword_result = self._bm25_search(query, limit)
        keyword_scores = [x[1] for x in keyword_result]
        print(keyword_result)
        
        semantic_result = self.semantic_search.search_chunks(query, limit)
        semantic_scores = [x['score'] for x in semantic_result]
        # print("...")
        # print(keyword_scores)
        # print(semantic_scores)
        # print("...")
        normalized_keyword_scores = normalize(keyword_scores)
        normalized_semantic_scores = normalize(semantic_scores)
        # print(normalized_keyword_scores)
        # print(normalized_semantic_scores)
        print(self.documents)
        scores_dict = {}
        #print(self.documents[1]['title'])
        for result, normalized_result in zip(keyword_result, normalized_keyword_scores):
            scores_dict[result[0]] = {
              'title' : self.documents[result[0]-1]['title'],
              'id' : self.documents[result[0]-1]['id'],
              'bm25_score' : result[1],
              'normalized_bm25_score' : normalized_result,
              'semantic_score' : 0.0,
              'normalized_semantic_score' : 0.0,
              'hybrid_score' : 0.0,
              'description' : self.documents[result[0]-1]['description'],
            }
        
        

        for result, normalized_result in zip(semantic_result, normalized_semantic_scores):
            # print(result)
            # print(normalized_result)
            # print(result['id'])
            if result['id'] in scores_dict:
                scores_dict[result['id']]['semantic_score'] = result['score']
                scores_dict[result['id']]['normalized_semantic_score'] = normalized_result
            
            else:
                scores_dict[result['id']] = {
                'title' : self.documents[result['id']]['title'],
                'id' : self.documents[result['id']]['id'],
                'bm25_score' : 0.0,
                'normalized_bm25_score' : 0.0,
                'semantic_score' : result['score'],
                'normalized_semantic_score' : normalized_result,
                'hybrid_score' : 0,
                'description' : self.documents[result[0]-1]['description'],
            }

            for key in scores_dict:
                scores_dict[key]['hybrid_score'] = hybrid_score(scores_dict[key]['normalized_bm25_score'], scores_dict[key]['normalized_semantic_score'], alpha)
            
        sorted_scores_dict = sorted(scores_dict.values(), key=lambda item: item['hybrid_score'], reverse=True)
        print()
        print(sorted_scores_dict)
        return sorted_scores_dict
            
        # scores_dict[ = { ]
        #     
        #     
        #     
        #     'semantic_score' : 
        #     'normalized_semantic_score' : 
        #     'hybrid_score' : 
        # }
        

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")