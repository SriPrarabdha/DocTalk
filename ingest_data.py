from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle

loader = UnstructuredFileLoader("")
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(raw_documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents , embeddings)

with open("vectorstore.pkl" , "wb") as f:
    pickle.dump(vectorstore , f)
