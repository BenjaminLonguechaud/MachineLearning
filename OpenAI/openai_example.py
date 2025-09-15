def summarize_text(text, api_key, model="gpt-3.5-turbo", max_tokens=100, temperature=0.5):
    """
    Summarize the input text using the OpenAI API.
    Args:
        text (str): The text to summarize.
        api_key (str): Your OpenAI API key.
        model (str): The model to use (default: gpt-3.5-turbo).
        max_tokens (int): Maximum number of tokens for the summary.
        temperature (float): Controls randomness/creativity (default: 0.5).
    Returns:
        str: The summary text.
    """
    client = OpenAI(api_key=api_key)
    if model == "gpt-3.5-turbo":
        prompt = f"Summarize the following text:\n{text}"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    else:
        prompt = f"Summarize the following text:\n{text}"
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].text.strip()
# openai_example.py

from openai import OpenAI

def generate_text(prompt, api_key, model="gpt-3.5-turbo", max_tokens=100):
    """
    Generate text from OpenAI given a prompt (compatible with openai>=1.0.0).
    Args:
        prompt (str): The input prompt for the model.
        api_key (str): Your OpenAI API key.
        model (str): The model to use (default: davinci-002).
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Controls randomness/creativity (default: 1.0).
    Returns:
        str: The generated text.
    """
    temperature = 1.0  # Default value, can be changed by user
    import inspect
    # Check if temperature is passed as a keyword argument
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    if 'temperature' in values:
        temperature = values['temperature']
    if model == "gpt-3.5-turbo":
        # Use chat endpoint for chat models
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    else:
        # Use completions endpoint for legacy completion models like davinci-002
        client = OpenAI(api_key=api_key)
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].text.strip()
