from abc import ABC, abstractmethod
from dataclasses import dataclass
from langchain_core.documents import Document


@dataclass
class RetrievalResult:
    document: Document   # has page_content and metadata (source, chunk_index)
    score: float


class RetrieverBase(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievalResult]:
        ...

    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        ...
