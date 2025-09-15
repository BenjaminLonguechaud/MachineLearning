from pyexpat import model
import openai
import config

openai.api_key = config.api_key

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
def summarize_text(text, api_key, model="gpt-3.5-turbo", max_tokens=100, temperature=0.5):
    client = openai.OpenAI()
    prompt = f"Summarize the following text:\n{text}"
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