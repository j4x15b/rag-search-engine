
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L6-v2"
        )

    def rerank(self, query, results, documents, limit):
        pairs = []

        for result in results:
            doc = documents[result["id"] - 1]

            text = (
                f"Titel: {doc['title']}\n"
                f"Beschreibung: {doc['description']}"
            )

            pairs.append((query, text))

        scores = self.model.predict(pairs)

        reranked = []

        for result, score in zip(results, scores):
            item = result.copy()
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:limit]