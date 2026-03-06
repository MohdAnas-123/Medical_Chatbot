# 🩺 AI Medical Assistant Chatbot

An intelligent, context-aware medical chatbot built with **LangChain**, **Flask**, and **Google Gemini**. This application uses Retrieval-Augmented Generation (RAG) to provide accurate medical information based on a curated vector database, complete with conversational memory for follow-up questions.

![Live Demo Status](https://img.shields.io/badge/Status-Live-success)
![Python Version](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3.x-green)

### 🚀 Live Demo
**[Click here to chat with the Medical Assistant!](https://medical-chatbot-gm63.onrender.com)** *(Note: The server may take 30-50 seconds to wake up if it has been inactive).*

## ✨ Key Features
* **Retrieval-Augmented Generation (RAG):** Answers are grounded in actual medical data, reducing AI hallucinations.
* **Conversational Memory:** Uses LangChain's `history_aware_retriever` to remember previous chat context and understand pronouns/follow-up questions.
* **Modern UI:** Responsive, dark-mode web interface built with HTML/CSS and jQuery.
* **Optimized Vector Search:** Powered by Pinecone and HuggingFace (`all-MiniLM-L6-v2`) embeddings.

## 🛠️ Technology Stack
* **Frontend:** HTML5, CSS3, Bootstrap 4, jQuery
* **Backend:** Python, Flask, Gunicorn
* **AI / LLM:** Google Gemini 1.5 Flash, LangChain
* **Embeddings:** HuggingFace Sentence Transformers
* **Database:** Pinecone Vector Database
* **Deployment:** Render

## 💻 How to Run Locally

## 1. **Clone the repository:**

   git clone [https://github.com/YOUR-USERNAME/medical-chatbot.git]
   
## 2. Install the dependencies:

   pip install -r requirements.txt
   
## 3. Set up Environment Variables:

   ## Create a .env file in the root directory and add your API keys:
   
   PINECONE_API_KEY=your_pinecone_key
   GOOGLE_API_KEY=your_google_key
   
##4. Run the Application:

   python app.py
   
  ## Open your browser and navigate to http://127.0.0.1:5000.
