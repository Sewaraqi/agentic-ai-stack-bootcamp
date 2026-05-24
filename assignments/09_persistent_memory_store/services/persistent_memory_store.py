import json
from pathlib import Path

from base.memory_base import MemoryBase, MemoryEntry
from services.embedding_service import EmbeddingService
from services.vector_memory_store import VectorMemoryStore


class PersistentMemoryStore(MemoryBase):
    def __init__(self, embedding_service: EmbeddingService, path: str = "data/memory.json") -> None:
        self._path = Path(path)
        self._vector_store = VectorMemoryStore(embedding_service)
        self._entries: list[MemoryEntry] = []

        if self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for d in data:
                self.add(d["role"], d["content"])

    def add(self, role: str, content: str) -> None:
        self._vector_store.add(role, content)
        self._entries.append(MemoryEntry(role=role, content=content))
        self._write()

    def search(self, query: str, top_k: int = 3) -> list[MemoryEntry]:
        return self._vector_store.search(query, top_k)

    def clear(self) -> None:
        self._vector_store.clear()
        self._entries.clear()
        self._path.unlink(missing_ok=True)

    def __len__(self) -> int:
        return len(self._entries)

    def _write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [{"role": e.role, "content": e.content} for e in self._entries]
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
