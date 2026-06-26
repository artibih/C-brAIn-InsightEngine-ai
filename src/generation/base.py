from abc import ABC, abstractmethod

class LLMResponseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, **kwargs) -> str:
        """Generate a response to the query, optionally using context documents"""
        pass

