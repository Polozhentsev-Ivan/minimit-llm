import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

from dotenv import load_dotenv
load_dotenv()

import torch
from langchain.schema import HumanMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class BaseModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        ...


class OpenAIModel(BaseModel):
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None, **client_kwargs: Any):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not provided")
        from langchain_community.chat_models import ChatOpenAI as _Chat

        self.client = _Chat(model_name=model_name, openai_api_key=api_key, **client_kwargs)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self.client([HumanMessage(content=prompt)]).content.strip()


class LlamaHFModel(BaseModel):
    def __init__(
        self,
        repo_id: str = "meta-llama/Meta-Llama-3-70B-Instruct",
        four_bit: bool = True,
        hf_token: Optional[str] = None,
        **model_kwargs: Any,
    ):
        hf_token = hf_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not provided")

        load_kwargs: Dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
        if four_bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(repo_id, token=hf_token, **load_kwargs)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str, **kwargs: Any) -> str:
        max_new     = int(kwargs.get("max_new_tokens", 48))
        do_sample   = bool(kwargs.get("do_sample", True))
        temperature = float(kwargs.get("temperature", 0.3))
        top_p       = float(kwargs.get("top_p", 0.95))

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_cfg = {"max_new_tokens": max_new,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else self.tokenizer.eos_token_id,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p}

        output_ids = self.model.generate(**inputs, **gen_cfg)[0]

        prompt_len = inputs["input_ids"].shape[-1]
        new_ids = output_ids[prompt_len:]
        
        completion = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if completion.startswith(prompt):
            completion = completion[len(prompt):].lstrip("\n")
        return completion

_MODEL_REGISTRY = {"openai": OpenAIModel, "llama_hf": LlamaHFModel}


def get_model(alias: str, **kwargs: Any) -> BaseModel:
    if alias not in _MODEL_REGISTRY:
        raise KeyError(alias)
    return _MODEL_REGISTRY[alias](**kwargs)
