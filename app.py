import os
import time
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Check Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found. Add your Hugging Face token to the .env file."
    )

# Create Hugging Face client
client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)

# Get topic from user
topic = input("Enter a topic: ").strip()

if not topic:
    print("Please enter a valid topic.")
    exit()

# Prompt
prompt = f"""
Create exactly 5 study flashcards about "{topic}".

For each flashcard:
1. Write a clear question.
2. Write a short and accurate answer.
3. Keep the difficulty suitable for a 2nd-year Computer Science student.
4. Focus on important concepts rather than trivial facts.
5. Keep each answer to 1-3 sentences.

Format the response exactly like this:

Flashcard 1
Question: ...
Answer: ...

Flashcard 2
Question: ...
Answer: ...

Flashcard 3
Question: ...
Answer: ...

Flashcard 4
Question: ...
Answer: ...

Flashcard 5
Question: ...
Answer: ...
"""

print("\nGenerating flashcards...\n")

# Measure response time
start_time = time.time()

try:
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Computer Science tutor "
                    "who creates high-quality study flashcards."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1500
    )

    end_time = time.time()

    latency = end_time - start_time

    # Get generated text
    response = completion.choices[0].message.content

    # Display result
    print("=" * 60)
    print("GENERATED FLASHCARDS")
    print("=" * 60)

    print(response)

    print("\n" + "=" * 60)
    print(f"Response time: {latency:.2f} seconds")
    print("=" * 60)

except Exception as e:
    print("\nError while generating flashcards:")
    print(e)