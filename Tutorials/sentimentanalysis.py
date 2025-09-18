from transformers import pipeline

sentiment_analysis = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """
    Analyze the sentiment of the given text using a pre-trained model.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary containing the sentiment label and score.
    """
    result = sentiment_analysis(text)[0]
    return {"label": result['label'], "score": result['score']}

input = "I love programming!"
output = analyze_sentiment(input)
print(input + " returns a " + output['label'] + " sentiment")

input = "I hate programming!"
output = analyze_sentiment(input)
print(input + " returns a " + output['label'] + " sentiment")
