import openai
import time
from openai.error import ServiceUnavailableError, RateLimitError, APIError


def call_chat_gpt(messages, model='gpt-3.5-turbo', stop=None, temperature=0.0, max_tokens=128, n=1):
    """Robust wrapper around the OpenAI ChatCompletion endpoint with retry‑backoff."""
    wait = 1
    while True:
        try:
            return openai.ChatCompletion.create(
                model=model,
                max_tokens=max_tokens,
                stop=stop,
                messages=messages,
                temperature=temperature,
                n=n,
            )
        except (ServiceUnavailableError, RateLimitError, APIError) as e:
            print('[OpenAI API] Retrying after exception:', e)
            time.sleep(min(wait, 60))
            wait *= 2


def chatgpt(messages, model='gpt-3.5-turbo', temperature=0.7, max_tokens=5500, stop=None):
    res = call_chat_gpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, stop=stop)
    return [c['message']['content'] for c in res['choices']]
