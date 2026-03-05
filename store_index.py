import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings

# 1. Load environment variables
# load_dotenv() automatically loads the variables into os.environ. 
# Langchain and Pinecone will find them automatically, so you don't need to reassign them.
load_dotenv()

# 2. Initialize Pinecone client and Embeddings
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index_name = "medical-chatbot"

# Load the embedding model (Make sure the dimension matches '384' used below)
embeddings = download_hugging_face_embeddings()

# 3. Check if the index exists, and handle accordingly
if not pc.has_index(index_name):
    
    # Create the index
    pc.create_index(
        name=index_name,
        dimension=384, # Make sure your HuggingFace model outputs 384 dimensions!
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    
    # ONLY load and split the PDF if we actually need to upload it
    extracted_data = load_pdf_file(data='data/')
    filter_data = filter_to_minimal_docs(extracted_data)
    text_chunks = text_split(filter_data)
    
    # Upload the documents to the new Pinecone index
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings, 
    )

else:
    
    # Do NOT use from_documents here. Use from_existing_index to just connect.
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )