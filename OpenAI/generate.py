from pyexpat import model
import openai
import config

openai.api_key = config.api_key

"""
Generate text from OpenAI given a prompt.
Args:
    prompt (str): The input prompt for the model.
    api_key (str): Your OpenAI API key.
    model (str): The model to use (default: gpt-3.5-turbo).
    max_tokens (int): Maximum number of tokens to generate.
Returns:
    str: The generated text.
"""
def generate_text(prompt, model="gpt-3.5-turbo", max_tokens=10):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)