import os, sys
os.path.dirname(sys.executable)

import config

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

def get_thales_documentation(user_input, api_key=config.api_key, model="gpt-3.5-turbo", max_tokens=100, temp=0.5):
    """
    Retrieve and answer questions about Thales CipherTrust documentation using LangChain and OpenAI.

    This function loads documentation from the Thales CipherTrust Platform website, splits it into manageable chunks, embeds the text, stores it in a vector database, and sets up a conversational retrieval chain to answer user questions.

    Args:
        user_input (str): The user's question about the documentation.
        api_key (str): OpenAI API key for authentication.
        model (str, optional): The OpenAI model to use (default: "gpt-3.5-turbo").
        max_tokens (int, optional): Maximum number of tokens for the answer (default: 100).
        temperature (float, optional): Sampling temperature for response creativity (default: 0.5).

    Returns:
        None. Prints the answer to the user's question.
    """
    # 1. Define the documentation URL
    url = "https://thalesdocs.com/ctp/"

    # 2. Load the documentation from the web
    loader = WebBaseLoader(url)
    raw_documents = loader.load()

    print(f"Number of documents: {len(raw_documents)}")

    # 3. Create a prompt template for contextualizing the search query
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("ai", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"),
        ]
    )

    # 4. Split the loaded documents into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(raw_documents)

    # 5. Create embeddings for the document chunks using OpenAI
    embeddings = OpenAIEmbeddings(api_key)

    # 6. Store the embeddings in a Facebook AI Similarity Search (FAISS) vector database for efficient retrieval
    vector_store = FAISS.from_documents(documents, embeddings)

    # 7. Set up a conversation memory buffer to maintain chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(model=model, temperature=temp, openai_api_key=api_key, max_tokens=max_tokens)

    # 8. Create a history-aware retriever for context-aware search
    history_aware_retriever = create_history_aware_retriever(llm, vector_store.as_retriever(), contextualize_q_prompt)

    # 9. Create a prompt template for answering questions based on retrieved context
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, contextualize_prompt)

    # 10. Create a retrieval-augmented generation (RAG) chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 11. Initialize chat history and invoke the chain with the user's input
    chat_history = []
    response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=user_input), AIMessage(content=response["answer"])])
    return response["answer"]


prompt = "What are the steps to install CipherTrust Platform?"
result = get_thales_documentation(prompt, config.api_key)
print(result["answer"])
