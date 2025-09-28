from dataclasses import dataclass, field
from typing import Optional

from .prompt_builder import SYSTEM_PROMPT, build_prompt
from .llm_client import LLMClient
from .storage import save_to_jsonl

@dataclass
class Configurations:
    provider: str         = "openai"
    model: str            = "gpt-5"
    temperature: float    = 0.0
    reasoning_effort: str = "low"
    system_prompt: str    = SYSTEM_PROMPT

@dataclass
class ApiKeys:
    openai: str | None = None
    gemini: str | None = None

class JDWorker:

    # ------ Initialize the worker ------
    def __init__(self, config: Optional[Configurations] = None):
        self.config: Configurations = config or Configurations()
        self._keys = ApiKeys()
        self._llm: Optional[LLMClient] = None
        self._need_rebuild = True

    # ------ Configuration change ------
    def set_provider(self, provider: str):
        if provider != self.config.provider:
            self.config.provider = provider
            self._need_rebuild = True

    def set_api_key(self, provider: str, key: Optional[str]):
        p = (provider or "").lower()
        if p == "openai":
            self._keys.openai = key
            if self.config.provider == "openai":
                self._need_rebuild = True
        elif p == "gemini":
            self._keys.gemini = key
            if self.config.provider == "gemini":
                self._need_rebuild = True
        else:
            raise ValueError("provider must be 'openai' or 'gemini'")

    def set_model(self, model: str):
        self.config.model = model
        if self._llm:
            self._llm.set_model(model)

    def set_temperature(self, temperature: float):
        self.config.temperature = float(temperature)
        if self._llm:
            self._llm.set_temperature(self.config.temperature)

    def set_reasoning(self, reasoning_effort: str):
        self.config.reasoning_effort = reasoning_effort
        if self._llm:
            self._llm.set_reasoning_effort(self.config.reasoning_effort)

    def set_system_prompt(self, system_prompt: str):
        self.config.system_prompt = system_prompt
        if self._llm:
            self._llm.set_system_prompt(system_prompt)

    # ------ Actions ------
    def generate(self, jd_text: str) -> str:
        llm = self._ensure_client()
        llm.set_system_prompt(self.config.system_prompt)
        user_prompt = build_prompt(jd_text)
        return llm.query(user_prompt)

    def save(self, ai_text: str, path: str) -> None:
        save_to_jsonl(ai_text, path)

    # ------ Buildup/Rebuild LLM Client ------
    def _ensure_client(self) -> LLMClient:
        if self._llm is None or self._need_rebuild:
            if self.config.provider == "openai":
                self._llm = LLMClient.init_openai_cilent(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    reasoning_effort=self.config.reasoning_effort,
                    api_key=self._keys.openai,
                )
            else:  # "gemini"
                self._llm = LLMClient.init_gemini_client(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    reasoning_effort=self.config.reasoning_effort,
                    api_key=self._keys.gemini,
                )
            self._llm.set_system_prompt(self.config.system_prompt)
            self._need_rebuild = False
        return self._llm