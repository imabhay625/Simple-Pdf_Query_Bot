import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS  # Importing FAISS from the correct module
FAISS.allow_dangerous_deserialization = True
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
import joblib

FAISS.allow_dangerous_deserialization = True
pickle_file_path = "faiss_index.pkl"

# Check if the file exists before opening it
if os.path.exists(pickle_file_path):
    # Now load the pickle file
    with open(pickle_file_path, "rb") as f:
        new_db = FAISS.load_local(f, allow_dangerous_deserialization=True)



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    #FAISS.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

FAISS.allow_dangerous_deserialization = True

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Set allow_dangerous_deserialization to True
    FAISS.allow_dangerous_deserialization = True


    


    # Now load the pickle file
    #new_db = joblib.load("faiss_index.joblib")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    if response and "output_text" in response:
        st.write("Reply: ", response["output_text"])
        return response["output_text"]
    else:
        st.write("Reply: Answer is not available.")
        return None

def main():
    memory = []

    st.set_page_config("Hey..! Research Papers 🧠")
    st.header("Ask Question Based on Research Papers 🧠")

    user_question = st.text_input("Ask a Question from the Research papers")

    if user_question:
        response = user_input(user_question)
        memory.append((user_question, response))

    with st.sidebar:
        st.title("Hey..")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit ", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Catching up..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done Below is your Raw text of Research Paper")
                st.write(raw_text)
    st.subheader("Memory:")
    for i, (question, answer) in enumerate(memory, start=1):
        st.write(f"{i}. Question: {question}")
        st.write(f"Answer: {answer}")
        st.write("-----")

if __name__ == "__main__":
    main()
