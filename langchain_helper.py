from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
load_dotenv()


cohere_llm = Cohere(cohere_api_key=os.environ["cohere_api_key"])


instructor_embeddings=HuggingFaceEmbeddings()
vectordb_filepath="faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column='prompt', encoding='latin-1')
    data = loader.load()
    vectordb=FAISS.from_documents(documents=data,embedding=instructor_embeddings)
    vectordb.save_local(vectordb_filepath)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_filepath, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=cohere_llm,
    chain_type="stuff",
    input_key="question",
    output_key="answer",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs = {"prompt": PROMPT})

    return chain

if __name__=='__main__':
    create_vector_db()
    chain=get_qa_chain()
    print(chain("do you provide internship> do you provide EMI option"))