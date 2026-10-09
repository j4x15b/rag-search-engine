import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
api_key_ollama = os.environ.get("OLLAMA_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
else: print("api_key_loaded")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key=api_key_ollama,  # Ollama ignores this, but the SDK requires a non-empty string
)

print("sending message")
completion = client.chat.completions.create(
  #model="gpt-6-astra",
  model="llama3.1:8b",
  messages = [
    {
        "role": "user",
        "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum. Use the internet to find information about boot.dev",
    }
    ]
)


print("response:")
print(completion.choices[0].message)
print("Tokens:")
#print(completion.usage)
print(f"Prompt tokens: {completion.usage.prompt_tokens}")
print(f"Response tokens: {completion.usage.completion_tokens}")
