from openai import OpenAI

from src.generation.base import GenerateResponse

MODEL = "gpt-4o-mini"

client = OpenAI()

with open("./src/generation/system_prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()


def generate_completion(prefix: str, suffix: str, *_args) -> GenerateResponse:
    completion = client.chat.completions.create(
        model=MODEL,
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
    return GenerateResponse(
        completion=completion.choices[0].message.content,
        input_tokens=completion.usage.prompt_tokens,
        output_tokens=completion.usage.completion_tokens,
        model=MODEL,
        cost=cost,
    )


if __name__ == "__main__":
    print(generate_completion("def add(a, b):", "a + b"))
