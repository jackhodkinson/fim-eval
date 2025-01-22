from typing import Tuple

from openai import OpenAI

client = OpenAI()

with open("./src/generation/system_prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()


def generate_completion(prefix: str, suffix: str) -> Tuple[str, float]:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<PRE>{prefix}<SUF>{suffix}<MID>"},
        ],
    )

    input_token_cost = 0.150 / 1000000
    output_token_cost = 0.600 / 1000000
    cost = (
        completion.usage.completion_tokens * output_token_cost
        + completion.usage.prompt_tokens * input_token_cost
    )
    return completion.choices[0].message.content, cost


if __name__ == "__main__":
    print(generate_completion("def add(a, b):", "a + b"))
