from abc import ABC, abstractmethod
from dataclasses import dataclass
from langchain_core.documents import Document


@dataclass
class RetrievalResult:
    document: Document
    score: float


class RetrieverBase(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievalResult]:
        ...

    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        ...
