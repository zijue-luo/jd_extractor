from openai import OpenAI
from google import genai
from google.genai import types
import os

class LLMClient:
    def __init__(
        self,
        provider: str,
        client_obj,
        model: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        reasoning_effort: str = "low"
    ):
        # Basic fields
        self.provider = provider                  # "openai" | "gemini"
        self.client = client_obj                  # OpenAI: OpenAI(); Gemini: genai.Client
        self.model = model                        # model name
        self.system_prompt = system_prompt or ""  # optional system prompt
        self.temperature = float(temperature)     # 0.0 ~ 1.0
        # ["high","medium","low","minimal","dynamic"]; "dynamic" is for Gemini thinking
        self.reasoning_effort = reasoning_effort

    # ---------------- OpenAI ----------------
    @classmethod
    def init_openai_client(  # keep your original name
        cls,
        model: str,
        api_key=None,
        system_prompt: str = "",
        temperature: float = 0.0,
        reasoning_effort: str = "low"
    ):
        """Initialize an OpenAI chat client."""
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Missing OPENAI_API_KEY")
        client = OpenAI(api_key=key)
        return cls(
            provider="openai",
            client_obj=client,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            reasoning_effort=reasoning_effort
        )

    # ---------------- Gemini (google-genai) ----------------
    @classmethod
    def init_gemini_client(
        cls,
        model: str,
        api_key=None,
        system_prompt: str = "",
        temperature: float = 0.0,
        reasoning_effort: str = "low"
    ):
        """Initialize a Gemini client (google-genai)."""
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("Missing GEMINI_API_KEY")
        client = genai.Client(api_key=key)
        return cls(
            provider="gemini",
            client_obj=client,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            reasoning_effort=reasoning_effort
        )

    # ---------------- Query ----------------
    def query(self, prompt: str) -> str:
        """Send prompt and get plain text response."""
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort
            )
            return (resp.choices[0].message.content or "").strip()

        elif self.provider == "gemini":
            text = (self.system_prompt + "\n" + prompt).strip() if self.system_prompt else prompt

            # Always map reasoning_effort -> thinking_budget; if None, don't pass it.
            budget = self._get_thinking_budget()
            thinking_cfg = types.ThinkingConfig(thinking_budget=budget) if isinstance(budget, int) else None

            gen_config = types.GenerateContentConfig(
                temperature=self.temperature,
                thinking_config=thinking_cfg  # None is fine
            )

            resp = self.client.models.generate_content(
                model=self.model,
                contents=text,
                config=gen_config
            )
            content = resp.text or ""
            return content.strip()

        else:
            raise NotImplementedError(f"Unsupported provider: {self.provider}")

    # ---------------- Helpers ----------------
    def _get_thinking_budget(self):
        """Map reasoning_effort to Gemini thinking_budget."""
        mapping = {
            "dynamic": None,
            "minimal": 128,
            "low": 512,
            "medium": 2048,
            "high": 8192
        }
        key = (self.reasoning_effort or "").lower()
        return mapping.get(key, None)

    # ---------------- Setters ----------------
    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt or ""

    def set_temperature(self, temperature: float):
        self.temperature = float(temperature)

    def set_reasoning_effort(self, reasoning_effort: str):
        self.reasoning_effort = reasoning_effort

    def set_model(self, model: str):
        self.model = model


# For debugging and an example to use
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_provider = "openai"
    if test_provider == "openai":
        MODEL_NAME = "gpt-5"
        llm = LLMClient.init_openai_client(model=MODEL_NAME, api_key=API_KEY, temperature=0.0, reasoning_effort="low")
    elif test_provider == "gemini":
        MODEL_NAME = "gemini-2.5-flash"
        llm = LLMClient.init_gemini_client(model=MODEL_NAME, api_key=API_KEY, temperature=0.0, reasoning_effort="low")

    user_prompt = """
    Hello, World!
    """

    try:
        print(f"[INFO] Using model {MODEL_NAME}")
        result_text = llm.query(user_prompt)
        print(f"\b=== LLM Raw Output ===")
        print(result_text)
    except Exception as e:
        print(f"[Error] {e}")