from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def sentiment_analysis(text, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Analyze the sentiment of the given text using a pre-trained model.

    Args:
        text (str): The input text to analyze.
        model_name (str, optional): The name of the pre-trained model to use for sentiment analysis (default: "distilbert-base-uncased-finetuned-sst-2-english").

    Returns:
        dict: A dictionary containing the sentiment label and score.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids_pt = tokenizer(text, return_tensors="pt")
    print(input_ids_pt)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    with torch.no_grad():
        outputs = model(**input_ids_pt)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
    labels = model.config.id2label

    sentiment = {labels[i]: probabilities[i] for i in range(len(labels))}
    print(sentiment)
    return sentiment


input = "I love programming!"
output = sentiment_analysis(input)
input = "I hate programming!"
output = sentiment_analysis(input)