import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

from dotenv import load_dotenv
load_dotenv()

import torch
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline


class BaseModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        ...


class OpenAIModel(BaseModel):
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None, **client_kwargs: Any):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not provided")
        self.client = ChatOpenAI(model_name=model_name, openai_api_key=api_key, **client_kwargs)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self.client([HumanMessage(content=prompt)]).content


class LlamaHFModel(BaseModel):
    def __init__(
        self,
        repo_id: str = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        four_bit: bool = True,
        hf_token: Optional[str] = None,
        **pipeline_kwargs: Any,
    ):
        hf_token = hf_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not provided")

        model_kwargs: Dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
        if four_bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, use_auth_token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(repo_id, use_auth_token=hf_token, **model_kwargs)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, **pipeline_kwargs)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        gen_cfg = {
            "max_new_tokens": kwargs.get("max_new_tokens", 128),
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
        }
        return self.generator(prompt, **gen_cfg)[0]["generated_text"]

