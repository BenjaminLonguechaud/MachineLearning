from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import requests

try:
    url = input("https://en.wikipedia.org/wiki/Thales_Group")
    response = requests.get(url, timeout=10)
    webpage_text = response.text
    print("Webpage text loaded.")
except requests.exceptions.Timeout:
    print("Request timed out.")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

def faq_bot(question):

    # 1. Load the pre-trained BERT model and tokenizer for question answering
    model = BertForQuestionAnswering.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    context = webpage_text
    input_ids = tokenizer.encode(question, context)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_idx + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    start_index = torch.argmax(output.start_logits)
    end_index = torch.argmax(output.end_logits)
    if start_index <= end_index:
        answer = "".join(tokens[start_index : end_index + 1])
    else:
        answer = "I am unable to find the answer to your question."
    print("Answer: ", answer)

question = "When was the Thales Group founded?"

