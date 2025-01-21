from openai import OpenAI

client = OpenAI()

with open("./src/generation/system_prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()


def generate_completion(prefix: str, suffix: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<PRE>{prefix}<SUF>{suffix}<MID>"},
        ],
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    print(generate_completion("def add(a, b):", "a + b"))
