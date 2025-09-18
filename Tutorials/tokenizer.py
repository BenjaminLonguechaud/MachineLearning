from transformers import AutoTokenizer

def tokenize_text(text, model_name="bert-base-uncased"):
    """
    Tokenize the given text using a pre-trained tokenizer.

    Args:
        text (str): The input text to tokenize.
        model_name (str, optional): The name of the pre-trained model to use for tokenization (default: "bert-base-uncased").

    Returns:
        dict: A dictionary containing the tokenized input IDs and attention mask.
    """
    print("All Ids, types and masks from the input text: ")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens_id_maks = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    print(tokens_id_maks)

    print("Input tokens: ")
    tokens = tokenizer.tokenize(text)
    print(tokens)

    print("Ids of each input token: ")
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(token_ids)

    print("Decoded ids: ")
    decoded_ids = tokenizer.decode(token_ids)
    print(decoded_ids)

    return tokens

input = "I love programming!"
output = tokenize_text(input)
