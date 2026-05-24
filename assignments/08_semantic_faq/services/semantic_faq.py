from dataclasses import dataclass

from services.embedding_service import EmbeddingService


@dataclass
class SemanticFaqConfig:
    threshold: float = 0.70


class SemanticFaq:
    def __init__(self, config: SemanticFaqConfig, embedding_service: EmbeddingService):
        self._config = config
        self._embedding_service = embedding_service
        self._index: list[tuple[list[float], str, str]] = []  # (vector, question, answer)

    def load(self, pairs: list[tuple[str, str]]) -> None:
        self._index = []
        for question, answer in pairs:
            vector = self._embedding_service.embed(question)
            self._index.append((vector, question, answer))

    def add(self, question: str, answer: str) -> None:
        vector = self._embedding_service.embed(question)
        self._index.append((vector, question, answer))

    def _score_all(self, query_vector: list[float]) -> list[tuple[float, str, str]]:
        results = [
            (self._embedding_service.similarity(query_vector, vector), question, answer)
            for vector, question, answer in self._index
        ]
        return sorted(results, key=lambda t: t[0], reverse=True)

    def ask(self, query: str) -> str:
        query_vector = self._embedding_service.embed(query)
        ranked = self._score_all(query_vector)
        if not ranked:
            return "No matching FAQ entry found. (best score: 0.00)"
        best_score, _, best_answer = ranked[0]
        if best_score >= self._config.threshold:
            return best_answer
        return f"No matching FAQ entry found. (best score: {best_score:.2f})"

    def top_matches(self, query: str, n: int = 3) -> list[tuple[float, str, str]]:
        query_vector = self._embedding_service.embed(query)
        return self._score_all(query_vector)[:n]
