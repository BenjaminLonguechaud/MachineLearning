
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

def remove_hash_signs(text):
    """
    Remove all '#' characters from the input string and return the result.

    Args:
        text (str): The input string.

    Returns:
        str: The string with all '#' characters removed.
    """
    return text.replace('#', '')

def bert_embeddings(question, answer):

    """
    Compute and display the answer span from a context using a BERT model fine-tuned for question answering.

    Args:
        question (str): The question to ask.
        answer (str): The context or passage containing the answer.
    """
    # 1. Load the pre-trained BERT model and tokenizer for question answering
    model = BertForQuestionAnswering.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 2. Tokenize and encode the question and context
    encoding = tokenizer.encode_plus(question, answer)
    print(encoding)

    # 3. Extract input IDs and token type IDs from the encoding
    input_ids = encoding["input_ids"]
    token_type_ids = encoding["token_type_ids"]
    # attention_mask = encoding["attention_mask"]

    # 4. Convert input IDs to tokens for interpretability
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(tokens)

    # 5. Pass the encoded inputs to the model to get start and end logits
    output = model(
        input_ids=torch.tensor([input_ids]),
        token_type_ids=torch.tensor([token_type_ids])
    )

    # 6. Find the most likely start and end positions for the answer
    start_index = torch.argmax(output.start_logits)
    end_index = torch.argmax(output.end_logits)

    # 7. Extract and print the answer span from the tokens
    answer = "".join(tokens[start_index : end_index + 1])
    answer = remove_hash_signs(answer)
    print("Answer: ", answer)


question = "What is your name?"
answer = "My name is ChatGPT. I am an AI language model developed by OpenAI."
bert_embeddings(question, answer)