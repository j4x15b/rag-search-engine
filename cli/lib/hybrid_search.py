import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies, DEFAULT_TEXT_LENGTH

######################################################################################
# CLI FUNCTIONS
#
######################################################################################

#Command Method for CLI input weighted-search
def weighted_search_command(query, alpha=0.5, limit=5):
    movie_list = load_movies()
    working_limit = min(limit*500, len(movie_list))
    hybrid_search = HybridSearch(movie_list)
    
    results = hybrid_search.weighted_search(query, alpha, limit)
    for i, result in enumerate(results[:limit]):
            print(f"\n{i+1}. {result['title']}")
            print(f" BM25: {result['normalized_bm25_score']:.3f}, Semantic: {result['normalized_semantic_score']:.3f}")
            print(f" {result['description']}")

#Command Method for CLI input: rrf-search
def rrf_search_command(query:str, k, limit:int, enhance_method:str=None, rerank_method:str=None):

    movie_list = load_movies()
    hybrid_search = HybridSearch(movie_list)

    
    if enhance_method:
        query = llm_enhancement(query, enhance_method)

    #search continues
    print('rrf_search proceeds')
    if rerank_method == "individual":
        working_limit = limit *5
        results = hybrid_search.rrf_search(query, k, working_limit)    

        rank_list = []
        for document in results:
            movie_doc = hybrid_search.documents[document['id']]
            rank = llm_rerank(query, movie_doc)
            rank_list.append((rank, movie_doc))
            
            print(rank, movie_doc['title'])
        sorted_rank_list = sorted(rank_list, key=lambda item: item[0], reverse=True)
        print(sorted_rank_list)
        for i, result in enumerate(results[:limit]):
            print(f"\n{i+1}. {result['title']}")
            print(f" RRF Score: {result['rrf_score']}")
            print(f" BM25: {result['bm_25_rank']:.3f}, Semantic: {result['semantic_rank']:.3f}")
            print(f" {result['description']}")
    
    else: 
        results = hybrid_search.rrf_search(query, k, limit)
        print('rrf_search results')
        for i, result in enumerate(results[:limit]):
            print(f"\n{i+1}. {result['title']}")
            print(f" RRF Score: {result['rrf_score']}")
            print(f" BM25: {result['bm_25_rank']:.3f}, Semantic: {result['semantic_rank']:.3f}")
            print(f" {result['description']}")
    

#######################################################################################
# CALCULATING FUNCTIONS
#
#######################################################################################

#calculate normalized list
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
        
#calculate hybrid score
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

#calculate rrf score 
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

#######################################################################################
# LLM FUNCTIONS
#
#######################################################################################
def send_to_llm(prompt: str, model: str = 'ollama'):
    import os
    from dotenv import load_dotenv
    from openai import OpenAI
    
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    api_key_ollama = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    #else: print("api_key_loaded")

    if model == 'ollama':
        client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key=api_key_ollama,  # Ollama ignores this, but the SDK requires a non-empty string
                )
        used_model="llama3.1:8b"
    else:
        client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                )
        used_model="openrouter/free"
        #used_model="gpt-6-astra"

    completion = client.chat.completions.create(
        model = used_model,
        messages = [
        {
            "role": "user",
            "content": prompt,
        }
        ]
    )

    print(f"Model used: {completion.model}")
    print(f"Tokens: Prompt tokens: {completion.usage.prompt_tokens}, Response tokens: {completion.usage.completion_tokens}")
    answer = completion.choices[0].message.content
    # print(rank)
    
    return answer

def llm_rerank(query: str, movie_doc):
    #asks the llm to rerank the results 

    
    #Output ONLY the number in your response, no other text or explanation.
    # import os
    # from dotenv import load_dotenv
    # from openai import OpenAI
    
    # load_dotenv()
    # api_key = os.environ.get("OPENROUTER_API_KEY")
    # api_key_ollama = os.environ.get("OLLAMA_API_KEY")
    # if not api_key:
    #     raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    # else: print("api_key_loaded")


    # #use model: OPENROUTER FREE
    # # client = OpenAI(
    # #     base_url="https://openrouter.ai/api/v1",
    # #     api_key=api_key,
    # # )

    # #use model: OLLAMA
    # client = OpenAI(
    #     base_url="http://localhost:11434/v1",
    #     api_key=api_key_ollama,  # Ollama ignores this, but the SDK requires a non-empty string
    # )

    # if client.api_key == api_key:
    #     used_model="openrouter/free"
    #     #used_model="gpt-6-astra"
    # elif client.api_key == api_key_ollama:
    #     used_model="llama3.1:8b"
    # else: raise RuntimeError("No Model selected")

    # completion = client.chat.completions.create(
    #     model = f"{used_model}",
    
    #     messages = [
    #     {
    #         "role": "user",
    #         "content": prompt,
    #     }
    #     ]
    # )
    
    prompt = f"""Rate how well this movie matches the search query.
    
        Query: "{query}"
        Movie: {movie_doc['title']} - {movie_doc['description']}
    
        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness
    
        Rate 0-10 (10 = perfect match).
        Output ONLY the number in your response, no other text or explanation.

        Score:"""
    
    result = send_to_llm(prompt)
    return result

def llm_enhancement(query: str, enhance_method: str=None):
    #if enhanced method: load prompt from function, ask LLM to enhance
    #if not: return query as given
    if enhance_method:
        prompt=get_prompt(query, enhance_method)
        print(f"enhance: {enhance_method}")
    else: return query

    print('asking LLM to enhance query')
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    api_key_ollama = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    #else: print("api_key_loaded")


    #use model: OPENROUTER FREE
    # client = OpenAI(
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=api_key,
    # )

    #use model: OLLAMA
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key=api_key_ollama,  # Ollama ignores this, but the SDK requires a non-empty string
    )

    if client.api_key == api_key:
        used_model="openrouter/free"
        #used_model="gpt-6-astra"
    elif client.api_key == api_key_ollama:
        used_model="llama3.1:8b"
    else: raise RuntimeError("No Model selected")

    completion = client.chat.completions.create(
        model = f"{used_model}",
    
        messages = [
        {
            "role": "user",
            "content": prompt,
        }
        ]
    )
    
    enhanced_query = completion.choices[0].message.content    
    print(f"Model used: {completion.model}")
    print(f"Tokens: Prompt tokens: {completion.usage.prompt_tokens}, Response tokens: {completion.usage.completion_tokens}")
    print(f"Enhanced query ({enhance_method}): '{query}' -> '{enhanced_query}'\n")
    
    if enhance_method == "expand":
        query = f"{query} {enhanced_query}"
    else:
        query = enhanced_query
    
    return query

def get_prompt(query:str, enhance_method:str):
    if enhance_method == "spell":
        prompt = f"""Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words. 
                Preserve punctuation and capitalization unless a change is required for a typo fix. 
                If there are no spelling errors, or if you're unsure, output the original query unchanged. 
                Output only the final query text, nothing else. 
                User query: "{query}"
                """
        
    elif enhance_method == "rewrite":
        prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.
                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep the rewritten query concise (under 10 words)
                - It should be a Google-style search query, specific enough to yield relevant results
                - Don't use boolean logic

                Examples:
                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                If you cannot improve the query, output the original unchanged.
                Output only the rewritten query text, nothing else.

                User query: "{query}"
                """
    elif enhance_method == "expand":
        prompt = f"""Expand the user-provided movie search query below with related terms.
                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                Output only the additional terms; they will be appended to the original query.
                
                Examples:
                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                Important: Don't explain, what you did and why, just output your additional terms, 
                the expansion of the given user query!
                
                User query: "{query}"
                """
    else: 
        print("no enhancement-method given")
        prompt = """
                    Output only the query text, nothing else.
                    User query: '{query}'
                """
    return prompt
    
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

    def rrf_search(self, query: str, k: int, limit: int = 5) -> list[dict]:
        #safety_function: if #limit is higher than #documents: use #documents
        working_limit = min(limit*500, len(self.documents))

        #proceeding bm_25_search
        bm_25_results = self._bm25_search(query, working_limit)
        #proceeding chunked Semantic Search
        semantic_results = self.semantic_search.search_chunks(query, working_limit)
    
        score_mapping_to_rank_dict = {}
        for i, result in enumerate(bm_25_results):
            id = result[0]
            score_mapping_to_rank_dict[result[0]] = {
                'id' : id,
                'title' : self.documents[id-1]['title'],
                'bm_25_rank' : i+1,
                'semantic_rank' : 0,
                'description' : self.documents[id-1]['description'][:DEFAULT_TEXT_LENGTH],
            }

        for i, result in enumerate(semantic_results):
            if result['id'] in score_mapping_to_rank_dict:
                score_mapping_to_rank_dict[result['id']]['semantic_rank'] = i+1
            else:
                score_mapping_to_rank_dict[result['id']] = {
                'id' : result['id'],
                'title' : self.documents[result['id']-1]['title'],
                'bm_25_rank' : 0,
                'semantic_rank' : i+1,
                'description' : self.documents[result['id']-1]['description'][:DEFAULT_TEXT_LENGTH],
                }

        # import itertools
        # print(dict(itertools.islice(score_mapping_to_rank_dict.items(), 20)))

        for movie_id, ranks in score_mapping_to_rank_dict.items():
            bm_25_rrf_score = 0.0
            semantic_rrf_score = 0.0

            if ranks['bm_25_rank'] != 0.0:
                bm_25_rrf_score = rrf_score(ranks['bm_25_rank'])
            if ranks['semantic_rank'] != 0.0:
                semantic_rrf_score = rrf_score(ranks['semantic_rank'])
        
            ranks['rrf_score'] = bm_25_rrf_score + semantic_rrf_score

        sorted_score_mapping_to_rank = sorted_results = sorted(
            score_mapping_to_rank_dict.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
            )

        gesamt = 0
        return sorted_score_mapping_to_rank[:limit]