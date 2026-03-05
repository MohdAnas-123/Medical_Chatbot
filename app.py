import os
from flask import Flask, render_template, request
from dotenv import load_dotenv

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt, contextualize_q_system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

app = Flask(__name__)
load_dotenv() 

embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot" 

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite-preview')

# 1. Memory Reformulator Prompt
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(chatModel, retriever, contextualize_q_prompt)

# 2. Main Question Answering Prompt
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(chatModel, qa_prompt)

# 3. Final Conversational Chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
print("System Ready!")

# Global memory list (Note: For local testing only. In a real multi-user app, use session IDs)
chat_history = []

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    global chat_history
    msg = request.form.get("msg", "")
    
    if not msg:
        return "Please enter a valid question."

    try:
        # Pass input AND history to the chain
        response = rag_chain.invoke({
            "input": msg, 
            "chat_history": chat_history 
        })
        
        answer = response['answer']
        
        # Save memory for the next turn
        chat_history.append(HumanMessage(content=msg))
        chat_history.append(AIMessage(content=answer))
        
        # Keep memory short to save API costs
        if len(chat_history) > 10:
            del chat_history[0:2]
            
        return str(answer)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Sorry, I encountered an error. Please try again."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)