from abc import ABC, abstractmethod
from dataclasses import dataclass
from langchain_core.documents import Document

"""
Document comes from langchain_core.documents, what does it include?
has 2 fields : page_content (the content of the chunk)
metadata ( a dict)
DocumentStore.load_file populates metadata with source (filename) , chunck_index
"""


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
