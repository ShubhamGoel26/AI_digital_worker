from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
print("API Key:", os.getenv("OPENAI_API_KEY"))  # Debug print
client = OpenAI()
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")