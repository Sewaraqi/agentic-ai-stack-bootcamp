from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# A MemoryEntry holds one conversation turn: who said it, what they said, its vector (embedding)
@dataclass
class MemoryEntry:
    role: str  # 'user' or 'assistant'
    content: str
    embedding: list[float] = field(default_factory=list)
    # field(default_factory=list)
# gives each entry its own empty list
# by default if we wrote embedding: list[float] = [] all entries would share the same list object


# Any class that implements these three methods is a valid memory store -> in-RAM, Pinecone, SQL, Redis, whatever
class MemoryBase(ABC):
    @abstractmethod
    def add(self, role: str, content: str) -> None:
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> list[MemoryEntry]:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...
