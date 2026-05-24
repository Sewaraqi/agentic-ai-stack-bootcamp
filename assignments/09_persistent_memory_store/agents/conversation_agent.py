from base.agent_base import AgentBase
from services.llm_client import LlmClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ConversationAgent(AgentBase):
    _SYSTEM = "You are a helpful AI assistant specializing in agentic AI systems."

    def __init__(self, llm_client: LlmClient) -> None:
        self._llm = llm_client
        self._history: list[BaseMessage] = []
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._SYSTEM),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ])
        self._chain = self._llm.build_chain(prompt)

    def chat(self, user_input: str) -> str:
        response = self._chain.invoke({"history": self._history, "question": user_input})
        self._history.append(HumanMessage(content=user_input))
        self._history.append(AIMessage(content=response))
        return response

    def reset(self) -> None:
        self._history.clear()
