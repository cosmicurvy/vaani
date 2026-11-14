from __future__ import annotations

from typing import Any, Optional, Type
from pydantic import BaseModel 
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq

class GroqJudge(DeepEvalBaseLLM):
    """DeepEval judge adapter using the project's GROQ_API_KEY."""
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.model_name = model_name
        self.model = ChatGroq(model=model_name, temperature=0)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Any:
        if schema is not None:
            return self.model.with_structured_output(schema).invoke(prompt)
        response = self.model.invoke(prompt)
        return getattr(response, "content", str(response))

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Any:
        if schema is not None:
            return await self.model.with_structured_output(schema).ainvoke(prompt)
        response = await self.model.ainvoke(prompt)
        return getattr(response, "content", str(response))

    def get_model_name(self) -> str:
        return f"Groq/{self.model_name}"
