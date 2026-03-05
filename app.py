import os
from flask import Flask, render_template, request
from dotenv import load_dotenv

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt # Ensure system_prompt is correctly imported

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Initialize Flask App & Environment
app = Flask(__name__)
load_dotenv() # This automatically handles your GOOGLE_API_KEY and PINECONE_API_KEY

# 2. Setup Embeddings & Vector Store
print("Initializing Embeddings and connecting to Pinecone...")
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot" 

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 3. Setup LLM & Chain
print("Initializing Google Gemini Model...")
chatModel = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite-preview')

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
print("System Ready!")

# ----------------- FLASK ROUTES -----------------

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"]) # Usually, chat submissions only need POST
def chat():
    # Get the user's message from the frontend
    msg = request.form.get("msg", "")
    print(f"User Query: {msg}")
    
    # Check if the user sent an empty message
    if not msg:
        return "Please enter a valid question."

    # Added Error Handling for web robustness
    try:
        response = rag_chain.invoke({"input": msg})
        print(f"Bot Response: {response['answer']}")
        return str(response["answer"])
        
    except Exception as e:
        # If the API fails, the app won't crash. It will return a polite error to the user.
        error_msg = f"Sorry, I encountered an error while processing your request: {str(e)}"
        print(error_msg)
        return error_msg

# Added the run block so you can start the app directly via `python app.py`
if __name__ == '__main__':
    # debug=True automatically reloads the server when you save changes to this file!
    app.run(host="0.0.0.0", port=5000, debug=True)