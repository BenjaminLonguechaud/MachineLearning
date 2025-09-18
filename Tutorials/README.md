# Tutorials

This folder contains Python tutorials and example scripts for using Hugging Face Transformers and related NLP techniques.

## Contents

- `bert.py`: Demonstrates BERT-based question answering and string preprocessing.
- `faq.py`: Loads webpage text and answers questions using a BERT model.
- `sentimentanalysis.py`: Performs sentiment analysis using a pre-trained pipeline.
- `tokenizer.py`: Shows how to tokenize text and inspect token IDs and decoded tokens.
- `zeroshot.py`: Performs zero-shot classification using a pre-trained pipeline.

## Requirements

To run the scripts in this folder, you need:

- Python 3.8 or higher
- The following Python packages:
  - `torch`
  - `transformers`
  - `requests` (for `faq.py`)

You can install the main dependencies with:

```bash
pip install torch transformers requests
```

Some scripts may require additional dependencies. See the top of each script or its docstring for details.
