import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load .env
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# --- Initialize Flask ---
app = Flask(__name__)
CORS(app)

# --- 1. Load and prepare data (do this once) ---
data_file_path = os.path.join(os.path.dirname(__file__), "data", "academic_policy.txt")
loader = TextLoader(data_file_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# --- 2. Create embeddings + vectorstore ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# --- 3. Setup the LLM and prompt ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
If the answer is not found in the context, politely state that you don't have enough information.
Keep the answer concise and to the point.

Context: {context}

Question: {input}
""")

# --- 4. Build the retrieval chain ---
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- 5. Make a helper function ---
def get_rag_response(question):
    response = retrieval_chain.invoke({"input": question})
    answer = response["answer"]
    return answer

# --- 6. API endpoint for frontend ---
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "")
    if not question.strip():
        return jsonify({"reply": "Please enter a question."}), 400

    try:
        answer = get_rag_response(question)
        return jsonify({"reply": answer})
    except Exception as e:
        print("Error in RAG pipeline:", e)
        return jsonify({"reply": f"Error: {str(e)}"}), 500

# --- Root test route ---
@app.route("/", methods=["GET"])
def home():
    return "Backend is running!"

if __name__ == "__main__":
    app.run(port=5000, debug=True)
