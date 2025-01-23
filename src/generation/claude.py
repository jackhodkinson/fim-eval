import anthropic

from src.generation.base import GenerateResponse

MODEL = "claude-3-5-haiku-20241022"

client = anthropic.Anthropic()

with open("./src/generation/system_prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()


def generate_completion(prefix: str, suffix: str) -> GenerateResponse:
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"<PRE>{prefix}<SUF>{suffix}<MID>"},
        ],
    )

    input_token_cost = 0.80 / 1000000
    output_token_cost = 4.00 / 1000000
    cost = (
        message.usage.input_tokens * input_token_cost
        + message.usage.output_tokens * output_token_cost
    )
    return GenerateResponse(
        completion=message.content[0].text,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        model=MODEL,
        cost=cost,
    )


if __name__ == "__main__":
    print(generate_completion("def add(a, b):", "a + b"))
