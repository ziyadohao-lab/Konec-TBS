from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

def load_qa():

    loader = TextLoader("knowledge.txt")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./db"
    )

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        retriever=db.as_retriever()
    )

    return qa