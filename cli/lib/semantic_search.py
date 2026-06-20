from sentence_transformers import SentenceTransformer

# Load the model (downloads automatically the first time)


def verify_model():
    semantic_search = SemanticSearch()
    #model.encode(text)
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        