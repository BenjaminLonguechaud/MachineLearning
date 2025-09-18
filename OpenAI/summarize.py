
import openai
import config

openai.api_key = config.api_key

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
    client = openai.OpenAI(api_key=api_key)
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


def key_words(text, api_key, model="gpt-3.5-turbo", max_tokens=100, temperature=0.5):
    """
    Extract keywords from a given text using the OpenAI API.

    Args:
        text (str): The input text from which to extract keywords.
        api_key (str): Your OpenAI API key.
        model (str, optional): The OpenAI model to use (default: "gpt-3.5-turbo").
        max_tokens (int, optional): Maximum number of tokens for the response (default: 100).
        temperature (float, optional): Sampling temperature for response creativity (default: 0.5).

    Returns:
        str: A string containing the extracted keywords, typically comma-separated.
    """
    client = openai.OpenAI(api_key)
    prompt = f"Extract keywords from the following text:\n{text}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You will be provided with a block of text, and your tesk is to extract a list of keywords from it"
            },
            {
                "role": "user",
                "content": "This function uses the OpenAI API to summarize input text and works with both chat and completion models."
            },
            {
                "role": "assistant",
                "content": "function, OpenAI, summarize, chat, completion"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

prompt = "Learning a little each day adds up. Research shows that students who make learning a habit are more likely to reach their goals."
summarized_text = summarize_text(prompt)
print(summarized_text)
words_list = key_words(prompt)
print(words_list)