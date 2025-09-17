from transformers import pipeline

def zero_shot_classification(text, candidate_labels):
    """
    Perform zero-shot classification on the given text using a pre-trained model.

    Args:
        text (str): The input text to classify.
        candidate_labels (list): A list of candidate labels for classification.

    Returns:
        dict: A dictionary containing the classification label and score.
    """
    classifier = pipeline("zero-shot-classification")
    result = classifier(text, candidate_labels)
    return {"label": result['labels'][0], "score": result['scores'][0]}

input = "I love programming!"
labels = ["positive", "negative", "neutral"]
output = zero_shot_classification(input, labels)
print(output)
