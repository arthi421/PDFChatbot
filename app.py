import streamlit as st
import base64
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import threading

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up Hugging Face model (GPT-2)
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).to(device)

# Initialize FAISS with required components
embedding_function = HuggingFaceEmbeddings()
index = None  # Replace with your FAISS index implementation
docstore = None  # Replace with your document store implementation
index_to_docstore_id = None  # Replace with your index-to-docstore ID mapping
faiss_index = FAISS(embedding_function, index, docstore, index_to_docstore_id)

# Function to summarize a single PDF (with caching)
@st.cache_data(show_spinner=False)
def summarize_pdf(pdf_file):
    try:
        if pdf_file is None:
            return "Error: Empty file received."

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(pdf_file.read())

        if os.path.getsize(temp_path) == 0:
            return f"Error: Empty file uploaded ({pdf_file.name})."

        loader = PyPDFLoader(temp_path)
        docs = loader.load_and_split()
        chain = load_summarize_chain(model, chain_type="map_reduce", prompt="")  # Use GPT-2 model here
        summary = chain.run(docs)

        # Delete the temporary file
        os.remove(temp_path)

        return summary

    except Exception as e:
        return f"Error summarizing PDF: {pdf_file.name}. Error: {e}"

# Function to summarize PDFs using the adapted model (with caching)
@st.cache_data(show_spinner=False)
def summarize_pdfs_from_folder(pdfs_folder):
    summaries = []

    def summarize_thread(pdf_file):
        summary = summarize_pdf(pdf_file)
        summaries.append(summary)

    threads = []
    for pdf_file in pdfs_folder:
        thread = threading.Thread(target=summarize_thread, args=(pdf_file,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return summaries

# Main function to run Streamlit app
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with your PDF files :books:")
    user_query = st.text_input("Ask a question about your documents:")

    st.title("Multiple PDF Summarizer")
    pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if pdf_files:
        st.header("Preview PDFs:")
        for pdf_file in pdf_files:
            # Display PDF preview
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(pdf_file.read())

            with open(temp_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

            # Delete the temporary file
            os.remove(temp_path)

        if st.button("Generate Summary"):
            with st.spinner("Summarizing PDFs..."):
                try:
                    summaries = summarize_pdfs_from_folder(pdf_files)
                    for i, summary in enumerate(summaries):
                        st.write(f"Summary for PDF {i+1}:")
                        st.write(summary)
                except Exception as e:
                    st.error(f"Error summarizing PDFs: {e}")

    if user_query:
        try:
            # Placeholder for FAISS search
            relevant_documents = []  # Replace with your FAISS search logic
            
            # Placeholder for question answering
            qa_result = {}  # Replace with your QA model logic

            # Placeholder for GPT-2 response
            inputs = tokenizer.encode(user_query, return_tensors="pt").to(device)
            outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            st.write("Chatbot Response:")
            st.write(response.strip())
            
            if relevant_documents:
                st.write("Relevant Documents:")
                for doc in relevant_documents:
                    st.write(doc)  # Adjust how you display relevant documents based on your implementation

        except Exception as e:
            st.error(f"Error generating response: {e}")

if __name__ == '__main__':
    main()
