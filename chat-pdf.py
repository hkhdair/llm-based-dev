from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
# from langchain_community.llms import OpenAI
# from langchain.chat_models import ChatOpenAI #Importing chat models from langchain is deprecated
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

# Load environment variables
load_dotenv()

def process_text(text):
    # Split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    return knowledgeBase

def main():
    st.title("Chat with your PDF :book:")
    
    pdf = st.file_uploader('Upload your PDF Document', type='pdf')
    
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # Text variable will store the pdf text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Create the knowledge base object
        knowledgeBase = process_text(text)
        
        query = st.text_input('Ask a question to the PDF')
        cancel_button = st.button('Cancel')
        
        if cancel_button:
            st.stop()
        
        if query:
            docs = knowledgeBase.similarity_search(query)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", api_key=os.getenv("OPENAI_API_KEY"))
            chain = load_qa_chain(llm, chain_type='stuff')
            
            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)
                
            st.write(response)
            
            
if __name__ == "__main__":
    main()
    

'''
https://medium.com/@johnthuo/chat-with-your-pdf-using-langchain-f-a-i-s-s-and-openai-to-query-pdfs-e7bfde086155

next steps:
Add support for multiple file formats
- Add support for other file formats such as .docx, .txt, and .csv
- Log the chat
- Add support for saving the chat history to a file for instance a .txt file
- Implement Document Indexing techniques by use of libraries
such as Elasticsearch or Apache Solr
- Enhance question answering capabilities: Explore advanced question answering techniques, 
such as using transformer models like BERT or GPT, to improve the accuracy and comprehension of the system.
- Use a model that supports multiple languages, most notably some BERT models do support this.

'''