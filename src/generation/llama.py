from openai import OpenAI

from src.generation.base import GenerateResponse

MODEL = "codellama:7b-code"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # dummy key, required but not used
)

with open("./src/generation/system_prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()


def generate_completion(prefix: str, suffix: str) -> GenerateResponse:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<PRE>{prefix}<SUF>{suffix}<MID>"},
        ],
    )

    # Ollama doesn't provide token counts directly, so we'll estimate
    # or you can set these to 0 if you don't need them
    input_tokens = len(SYSTEM_PROMPT + prefix + suffix) // 4  # rough estimate
    output_tokens = len(response.choices[0].message.content) // 4  # rough estimate

    return GenerateResponse(
        completion=response.choices[0].message.content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=MODEL,
        cost=0.0,  # Ollama is free to use locally
    )


if __name__ == "__main__":
    print(generate_completion("def add(a, b):", "a + b"))
