from http.client import responses

import openai
from google import genai
from google.genai import types
import tiktoken

from scrt import OPENAI_KEY, GOOGLE_KEY

WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RESET = "\033[0m"

MAX_TOKENS = {
    "default": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "o1": 200000,
    "o1-mini": 128000,
    "o3-mini": 200000,
    "gemini-2.0-flash": 1048576,
    "gemini-2.0-flash-thinking-exp": 32767,
    "gemini-2.0-flash-lite-preview-02-05": 1048576,
    "learnlm-1.5-pro-experimental": 32000
}

model_owner = {
    "openai": ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"],
    "google": ["gemini-2.0-flash", "gemini-2.0-flash-thinking-exp", "gemini-2.0-flash-lite-preview-02-05",
               "learnlm-1.5-pro-experimental"]
}

DEFAULT_MODEL = "gpt-4o-mini"

def count_context_length(prompt: str, model: str = "default") -> int:
    if model not in MAX_TOKENS or model == "default":
        model = DEFAULT_MODEL
    if model in model_owner["google"]:
        # Use a different method for Google's models
        return len(prompt.split())  # Example: count words as tokens
    else:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(prompt))

def model_max_context_length(model: str) -> int:
    if model in MAX_TOKENS:
        return MAX_TOKENS[model]
    return MAX_TOKENS["default"]

def is_context_too_long(prompt: str, model: str = "default") -> bool:
    return count_context_length(prompt, model) > model_max_context_length(model)

def basic_prompt(prompt: str, role :str = "You are a helpful assistant.", temperature: float = 0.2,
                 model: str ="default", debug: bool = False) -> str:
    if model not in MAX_TOKENS or model == "default":
        model = DEFAULT_MODEL
    if is_context_too_long(prompt, model):
        raise ValueError("Prompt exceeds the maximum token limit.")

    if debug:
        print(f"-------------Model: {model}-------------")
        print(f"{PINK}ROLE:\n{role}{RESET}")
        print(f"{BLUE}PROMPT:\n{prompt}{RESET}")

    if model in model_owner["google"]:
        response = _basic_prompt_gemini(prompt, role, temperature, model)
    else:
        response = _basic_prompt_openai(prompt, role, temperature, model)


    if debug:
        print(f"{GREEN}RESPONSE:\n{response}{RESET}")
        print(f"---")
    return response

def _basic_prompt_openai(prompt: str, role: str, temperature: float, model: str) -> str:
    openai.api_key = OPENAI_KEY

    response_text = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": role},
            {
                "role": "user",
                "content": prompt,
                "temperature": temperature
            }
        ]
    )
    return response_text.choices[0].message.content

def _basic_prompt_gemini(prompt: str, role: str, temperature: float, model: str) -> str:
    client = genai.Client(api_key=GOOGLE_KEY)
    # Add the role to the prompt for context
    role_prompt = f"TASK: {role} \n---\nPROMPT: {prompt}"

    response = client.models.generate_content(
        model=model,
        contents=role_prompt,
        config=types.GenerateContentConfig(
            temperature=temperature
        )
    )
    return response.text