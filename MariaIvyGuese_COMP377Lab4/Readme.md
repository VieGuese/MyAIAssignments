# COMP-377 Lab Assignment #4 – Prototype an Agentic RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot using LangChain and Google Gemini API. The chatbot answers questions based on the content of a provided `.txt` document.

## Features

*   Loads and chunks text documents.
*   Embeds document chunks using Gemini embeddings.
*   Stores embeddings in a FAISS vectorstore for efficient retrieval.
*   Utilizes LangChain to build a RAG chain.
*   Generates context-aware answers using Google Gemini Pro.
*   Includes a custom prompt for improved relevance and tone.
*   Returns the answer along with the source context.

## Setup and Installation

1.  **Clone the repository (or create the folder structure manually):**

    ```bash
    mkdir COMP377Lab4
    cd COMP377Lab4
    mkdir data
    ```

2.  **Place the txt document:**

    Place the `academic_policy.txt` file (or any other `.txt` document to be used by the chatbot) inside the `data/` directory.

3.  **Create a `.env` file:**

    Create a file named `.env` in the root directory of the project (`COMP377Lab4/`) and add the Google API Key:

    ```
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    ```

    Replace `YOUR_GEMINI_API_KEY` with your actual Google Gemini API key. You can obtain one from [Google AI Studio](https://ai.google.dev/).

4.  **Install dependencies:**

    Navigate to the project's root directory in your terminal and install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the chatbot, execute the main Python script:

```bash
python COMP377Lab4.py
```

The chatbot will initialize, and you can then type your questions in the terminal. Type `exit` to quit the chatbot.

## Code Overview

*   `COMP377Lab4.py`: This is the main script containing the chatbot's logic.
    *   **Document Loading and Chunking:** Uses `TextLoader` to load the `.txt` file and `RecursiveCharacterTextSplitter` to divide it into manageable chunks.
    *   **Embeddings:** `GoogleGenerativeAIEmbeddings` is used to convert text chunks into numerical vector representations.
    *   **Vectorstore:** FAISS (`FAISS.from_documents`) is employed to store and efficiently search through the embedded document chunks.
    *   **LLM Integration:** `ChatGoogleGenerativeAI` connects to the Gemini Pro model for generating responses.
    *   **RAG Chain:** `create_stuff_documents_chain` and `create_retrieval_chain` from LangChain are used to construct the RAG pipeline, which retrieves relevant document chunks and passes them to the LLM for context-aware answer generation.
    *   **Custom Prompt:** A `ChatPromptTemplate` defines a custom prompt to guide the LLM's response, ensuring it stays within the provided context and maintains a concise tone.
    *   **`get_rag_response(question)` function:** This function takes a user question, invokes the RAG chain, and prints the generated answer along with the source document chunks that were used to formulate the answer.
*   `data/`: Directory containing the `academic_policy.txt` file, which serves as the knowledge base.
*   `requirements.txt`: Lists all Python packages necessary for the project.
*   `.env`: (Not committed to version control) Stores your Google API Key securely.

## Submission Notes

Ensure compress the `COMP377Lab4/` directory into a `.zip` file, excluding the `venv/` (virtual environment) folder.

