import os
from crewai import LLM
from dotenv import load_dotenv
import requests
from urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings (not recommended for production)
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

load_dotenv()

class LLMHelper:
    """
    Helper class to initialize a CrewAI LLM instance for different providers:
      - groq
      - gemini
      - ollama

    The initialization differs per provider. For example:
      - Groq: Requires an API key and a Groq‑specific model string.
      - Gemini: Can use either a Vertex AI credentials JSON (if provided) or an API key.
      - Ollama: Typically uses a local base URL.
    """
    def __init__(self, provider: str, config: dict = None):
        """
        Initialize the LLMHelper.
        
        Args:
            provider (str): One of "groq", "gemini", or "ollama".
            config (dict, optional): A dictionary of configuration parameters such as:
                - model: The model identifier for the provider.
                - api_key: API key if required.
                - vertex_credentials: For Gemini, a JSON string of Vertex credentials.
                - base_url: For Ollama, the local endpoint URL.
                - temperature: Temperature setting (default 0.7).
                - ... any additional parameters accepted by LLM.
        """
        self.provider = provider.lower()
        self.config = config or {}
        self.llm = self._create_llm_instance()

    def _create_llm_instance(self):
        temperature = self.config.get("temperature", 0.7)
        if self.provider == "groq":
            # For Groq, use a model string like "groq/llama-3.2-90b-text-preview"
            model = self.config.get("model", "groq/llama-3.3-70b-versatile")
            # API key can be provided in config or via environment variable GROQ_API_KEY.
            api_key = self.config.get("api_key") or os.environ.get("GROQ_API_KEY")
            return LLM(model=model, api_key=api_key, temperature=temperature, verify=False)
        elif self.provider == "gemini":
            # For Gemini, use a model string like "gemini/gemini-1.5-pro-latest"
            model = self.config.get("model", "gemini/gemini-1.5-pro-latest")
            # Otherwise, fall back on an API key (config or env variable GEMINI_API_KEY).
            api_key = self.config.get("api_key") or os.getenv("GEMINI_API_KEY")
            return LLM(model=model, api_key=api_key, temperature=temperature, verify=False)
        elif self.provider == "ollama":
            # For Ollama, use a model string like "ollama/llama3.1" and a local base URL.
            model = self.config.get("model", "ollama/llama3.2:3b")
            base_url = self.config.get("base_url", "http://cvrsecslmst3:11434")
            return LLM(model=model, base_url=base_url, temperature=temperature, verify=False)
        else:
            raise ValueError(f"Unsupported provider '{self.provider}'. Supported providers are: groq, gemini, ollama.")

    def get_llm(self):
        """
        Retrieve the initialized LLM instance.
        
        Returns:
            LLM: The CrewAI LLM instance for the selected provider.
        """
        return self.llm


# Example usage:
if __name__ == "__main__":
    # Example configuration for Gemini:
    gemini_config = {
        "model": "gemini/gemini-1.5-pro-latest",
        # Optionally, pass vertex_credentials or api_key:
        "api_key": os.getenv("GEMINI_API_KEY"),
        "temperature": 0.6,
    }
    llm_helper = LLMHelper(provider="gemini", config=gemini_config)
    llm_instance = llm_helper.get_llm()
    print(f"Initialized LLM instance for Gemini: {llm_instance}")

    # Similarly, you can initialize for Groq or Ollama:
    # groq_helper = LLMHelper(provider="groq", config={"api_key": os.getenv("GROQ_API_KEY")})
    # ollama_helper = LLMHelper(provider="ollama", config={"base_url": "http://localhost:11434"})
