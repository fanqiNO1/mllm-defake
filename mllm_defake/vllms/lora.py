from mllm_defake.vllms.vllms import OpenAICompat
from loguru import logger


class VLLMServedLoRA(OpenAICompat):
    def __init__(self, api_key: str, proxy: str | None = None, base_url: str | None = None):
        # Set default base URL if none is provided
        if base_url is None:
            logger.warning("No base URL provided for VLLMServedLoRA, assuming http://127.0.0.1:8000/v1")
            base_url = "http://127.0.0.1:8000/v1"

        # Initialize the parent class with the API key, proxy, and base URL
        super().__init__(api_key, proxy, base_url)
        # Query the model name from the API
        self.model_name = self.query_model_name()
        self.short_name = "vllm"
