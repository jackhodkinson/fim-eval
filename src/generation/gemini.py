import os
import time
from collections import deque
from datetime import datetime, timedelta

import google.generativeai as genai

from src.generation.base import GenerateResponse

MODEL = "gemini-1.5-flash"

with open("./src/generation/system_prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel(model_name=MODEL, system_instruction=SYSTEM_PROMPT)

# Add rate limiting queue and parameters
_request_times = deque(maxlen=15)  # Store timestamps of last 15 requests
_rpm_limit = 15  # 15 requests per minute


def _wait_for_rate_limit():
    """Ensure we don't exceed 15 requests per minute"""
    if len(_request_times) < _rpm_limit:
        _request_times.append(datetime.now())
        return

    # Check if oldest request is more than 1 minute ago
    oldest_request = _request_times[0]
    time_diff = datetime.now() - oldest_request
    if time_diff < timedelta(minutes=1):
        # Need to wait until we can make another request
        sleep_time = (timedelta(minutes=1) - time_diff).total_seconds()
        time.sleep(sleep_time)
    _request_times.append(datetime.now())


def generate_completion(prefix: str, suffix: str) -> GenerateResponse:
    _wait_for_rate_limit()  # Add rate limiting check
    response = model.generate_content(f"<PRE>{prefix}<SUF>{suffix}<MID>")

    input_token_cost = 0.075 / 1000000
    output_token_cost = 0.300 / 1000000
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count
    cost = (input_tokens * input_token_cost) + (output_tokens * output_token_cost)

    return GenerateResponse(
        completion=response.text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=MODEL,
        cost=cost,
    )


if __name__ == "__main__":
    print(generate_completion("def add(a, b):", "a + b"))
