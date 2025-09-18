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
    temperature (float): Controls randomness/creativity (default: 0.5).
Returns:
    str: The generated text.
"""
def generate_text(prompt, api_key, model="gpt-3.5-turbo", max_tokens=10, temperature=0.5):
    client = openai.OpenAI(api_key=api_key)
    if model == "gpt-3.5-turbo":
        prompt = f"Summarize the following text:\n{prompt}"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    else:
        prompt = f"Summarize the following text:\n{prompt}"
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].text.strip()

prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)