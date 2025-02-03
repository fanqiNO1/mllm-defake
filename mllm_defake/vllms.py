from abc import ABC, abstractmethod

import httpx
from openai import OpenAI

from mllm_defake.utils import encode_image_to_base64


class VLLM(ABC):
    def __init__(self):
        self.short_name = "unknown"

    @abstractmethod
    def infer(self, system_prompt: str, user_prompt: str, image_path: str) -> str:
        """
        This method sends the user prompt and image files (if any) to the Vision-LLM API and
        returns the model's response.

        @param system_prompt: A system prompt that gives context to the model.
        @param user_prompt: The question or task prompt provided by the user.
        @param image_path: A path-like object representing the image file to be sent to the API.

        @return: The model's response to the user prompt. Always a string.
        """
        pass


class OpenAICompat(VLLM):
    def __init__(self, api_key: str, proxy: str = None):
        super().__init__()
        self.api_key = api_key
        self.http_client = httpx.Client(proxy=proxy) if proxy else httpx.Client()
        self.client = OpenAI(api_key=api_key, http_client=self.http_client)
        self.model_name = "gpt-4o"
        self.short_name = "gpt4o"

    def infer_raw(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1000,
        )
        return response.choices[0].message.content

    def infer(self, system_prompt: str, user_prompt: str, image_path: str) -> str:
        image_url = encode_image_to_base64(image_path)
        response = self.infer_raw(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ]
        )
        return response

    def query_model_name(self) -> str:
        # Get the model name from the API
        response = self.client.models.list()
        return response.data[0].id


class GPT4o(OpenAICompat):
    def __init__(self, api_key: str, proxy: str = None):
        super().__init__(api_key, proxy)
        self.model_name = "gpt-4o"
        self.short_name = "gpt4o"


class GPT4oMini(OpenAICompat):
    def __init__(self, api_key: str, proxy: str = None):
        super().__init__(api_key, proxy)
        self.model_name = "gpt-4o-mini"
        self.short_name = "gpt4omini"


class Llama32VisionInstruct(OpenAICompat):
    def __init__(self, api_key: str, proxy: str = None, base_url: str = None):
        self.api_key = api_key
        self.http_client = httpx.Client(proxy=proxy) if proxy else httpx.Client()
        self.client = OpenAI(
            api_key=api_key,
            http_client=self.http_client,
            base_url=base_url if base_url else "http://127.0.0.1:8000",
        )
        self.model_name = self.query_model_name()
        self.short_name = "llama32vi"


# @deprecated(reason="No public API available")
class InternVL2Pro(VLLM):
    def __init__(self, api_key: str, url: str, proxy: str = None):
        super().__init__()
        self.api_key = api_key
        self.url = url
        self.client = httpx.Client(proxy=proxy) if proxy else httpx.Client()

    def infer(self, system_prompt: str, user_prompt: str, image_path: str) -> str:
        # Prepare the file data for the request
        files = [("files", open(image_path, "rb"))] if image_path else []

        # Prepare the data payload
        data = {
            "question": f"{system_prompt}\n\n{user_prompt}",
            "api_key": self.api_key,
        }

        try:
            # Send the POST request to the InternVL2 API using httpx client
            response = self.client.post(self.url, files=files, data=data)

            # Check if the request was successful
            if response.status_code == 200:
                return response.json().get(
                    "response", "No response key found in the JSON."
                )
            else:
                return f"Error: {response.status_code}, {response.text}"

        except httpx.RequestError as e:
            return f"Request error: {e}"


class Qwen2VL(VLLM):
    def __init__(
        self,
        api_key: str,
        url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        proxy: str = None,
    ):
        super().__init__()
        self.api_key = api_key
        self.http_client = httpx.Client(proxy=proxy) if proxy else httpx.Client()
        self.client = OpenAI(
            api_key=api_key, http_client=self.http_client, base_url=url
        )

    def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str,
        max_tokens: int = 700,
    ) -> str:
        image_url = encode_image_to_base64(image_path)
        response = self.client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class Llama32VisionCoT(OpenAICompat):
    def __init__(self, api_key: str, proxy: str = None, base_url: str = None):
        super().__init__(api_key, proxy)

        self.api_key = api_key
        self.http_client = httpx.Client(proxy=proxy) if proxy else httpx.Client()
        self.client = OpenAI(
            api_key=api_key,
            http_client=self.http_client,
            base_url=base_url if base_url else "http://127.0.0.1:8000",
        )
        self.model_name = self.query_model_name()
        self.short_name = "llama32vcot"

    def infer_raw(self, messages: list[dict[str, str]]) -> str:
        # This model does not support system prompts, so we replace it with role `user`.
        if messages[0]["role"] == "system":
            messages[0]["role"] = "user"
            messages[0]["content"] = [{"type": "text", "text": messages[0]["content"]}]
        return super().infer_raw(messages)


class QVQ(OpenAICompat):
    def __init__(self, api_key: str, proxy: str = None, base_url: str = None):
        super().__init__(api_key, proxy)

        self.api_key = api_key
        self.http_client = httpx.Client(proxy=proxy) if proxy else httpx.Client()
        self.client = OpenAI(
            api_key=api_key,
            http_client=self.http_client,
            base_url=base_url if base_url else "http://127.0.0.1:8000",
        )
        self.model_name = self.query_model_name()
        self.short_name = "qvq"

    def infer_raw(self, messages: list[dict[str, str]]) -> str:
        # This model does not support system prompts, so we replace it with role `user`.
        if messages[0]["role"] == "system":
            messages[0]["role"] = "user"
            messages[0]["content"] = [{"type": "text", "text": messages[0]["content"]}]
        return super().infer_raw(messages)
