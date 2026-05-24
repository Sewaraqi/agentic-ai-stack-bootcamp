from base.agent_base import AgentBase
from services.embedding_service import EmbeddingService
from services.semantic_faq import SemanticFaq, SemanticFaqConfig


class SemanticFaqAgent(AgentBase):

    def __init__(
        self,
        embedding_service: EmbeddingService,
        faq_pairs: list[tuple[str, str]],
        config: SemanticFaqConfig | None = None,
    ) -> None:
        self._faq = SemanticFaq(config or SemanticFaqConfig(), embedding_service)
        self._faq_pairs = faq_pairs
        self._faq.load(faq_pairs)

    def chat(self, user_input: str) -> str:
        lowered = user_input.lower()

        if lowered.startswith("top "):
            query = user_input[4:].strip()
            matches = self._faq.top_matches(query, n=3)
            if not matches:
                return "Index is empty."
            lines = []
            for rank, (score, question, answer) in enumerate(matches, 1):
                lines.append(f"  [{rank}] score={score:.3f}")
                lines.append(f"      Q: {question}")
                lines.append(f"      A: {answer}")
            return "\n".join(lines)

        if lowered.startswith("add "):
            rest = user_input[4:]
            if " | " not in rest:
                return "Usage: add <question> | <answer>"
            question, answer = rest.split(" | ", 1)
            self._faq.add(question.strip(), answer.strip())
            return f"Added: '{question.strip()}'"

        return self._faq.ask(user_input)

    def reset(self) -> None:
        self._faq.load(self._faq_pairs)
