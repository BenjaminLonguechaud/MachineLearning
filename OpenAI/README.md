
# OpenAI

This folder contains Python scripts and utilities for working with OpenAI's API, including text generation, summarization, keyword extraction, conversational retrieval, and creative chatbots.

## Main Scripts

- `config.py`: Stores the OpenAI API key (do not share this file publicly).
- `generate.py`: Generate text from a prompt using OpenAI's API.
- `summarize.py`: Summarize input text and extract keywords using OpenAI models.
- `langchain.py`: Advanced retrieval-augmented generation (RAG) and conversational search over documentation using LangChain and OpenAI.
- `poet.py`: A creative chatbot that responds as an eighteenth-century poet using OpenAI's API.

> Note: The `openai_example.py` file has been removed or replaced by more focused scripts.


## Requirements

To run the scripts in this folder, you need:

- Python 3.8 or higher
- An OpenAI API key (set in `config.py`)
- The following Python packages:
	- `openai`
	- `langchain`
	- `langchain-core`
	- `langchain-openai`
	- `requests`
	- `faiss-cpu` (for vector search in `langchain.py`)
	- `tiktoken` (for some LangChain/OpenAI integrations)

You can install the main dependencies with:

```bash
pip install openai langchain langchain-core langchain-openai requests faiss-cpu tiktoken
```

Some scripts may require additional dependencies. See the top of each script or its docstring for details.

For more details, see the docstrings in each script. Make sure to install the required dependencies before running the scripts.
