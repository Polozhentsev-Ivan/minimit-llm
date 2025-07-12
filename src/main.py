import os

import openai
from dotenv import load_dotenv

load_dotenv()


class LlmClient:

    def __init__(self, api_key: str | None = None, model_name: str = "gpt-3.5-turbo") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY was not found — place it .env or pass it as an argument")

        self.model_name = model_name
        self.client = openai.OpenAI(api_key=self.api_key)


    def create_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as exc:
            return f"Error occurred while request to API OpenAI: {exc}"



if __name__ == "__main__":
    client = LlmClient()
    print(client.create_response("Write down Cantor's Theorem"))
