import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Configure Google API Key
# Make sure to set GOOGLE_API_KEY in your .env file
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")

# --- 1. Load and chunk the document ---
# Dataset: academic_policy.txt from the data/ folder
data_file_path = "./data/academic_policy.txt"
loader = TextLoader(data_file_path)
documents = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"Loaded {len(documents)} document and split into {len(chunks)} chunks.")

# --- 2. Embed the chunks using Gemini embeddings ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# --- 3. Store them in a FAISS vectorstore ---
# Using FAISS as specified in the assignment
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Vectorstore created and populated with embeddings.")

# --- 4. Implement a LangChain RAG chain to retrieve relevant chunks ---
# Create a retriever from the vectorstore
retriever = vectorstore.as_retriever()

# --- 5. Use Gemini API to generate context-aware answers ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

# --- 6. Add at least one custom prompt to improve relevance or tone ---
# Custom prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
If the answer is not found in the context, politely state that you don't have enough information.
Keep the answer concise and to the point.

Context: {context}

Question: {input}
""")

# Create a document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create the retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- Function to get response and source context ---
def get_rag_response(question):
    response = retrieval_chain.invoke({"input": question})
    answer = response["answer"]
    source_documents = response["context"]

    print("\n--- Answer ---")
    print(answer)

    print("\n--- Source Context ---")
    for i, doc in enumerate(source_documents):
        print(f"Document {i+1}:\n{doc.page_content}\n")
    return answer, source_documents

# --- Example Usage ---
if __name__ == "__main__":
    print("RAG Chatbot initialized. Ask me questions about the academic policy.\n")
    while True:
        user_question = input("Your question (type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            print("Exiting chatbot. Goodbye!")
            break
        get_rag_response(user_question)