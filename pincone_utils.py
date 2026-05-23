from dotenv import load_dotenv
import os
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Initialize text splitter and embedding function
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300,length_function=len)
embedding=HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-mpnet-base-v2")

# Initialize pinecone vector store
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("ragindex")
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding,
)

def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    return text_splitter.split_documents(documents)


# -------------------------
# Index Document
# -------------------------

def index_document_to_pc(file_path: str, file_id: int) -> bool:
    try:
        splits = load_and_split_document(file_path)

        for split in splits:
            split.metadata["file_id"] = file_id

        vectorstore.add_documents(splits)

        return True

    except Exception as e:
        print(f"Error indexing document: {e}")
        return False


# -------------------------
# Delete Document
# -------------------------

def delete_doc_from_pc(file_id: int) -> bool:
    try:
        # Delete using metadata filter (Pinecone style)
        index.delete(filter={"file_id": file_id})

        print(f"Deleted all vectors with file_id {file_id}")
        return True

    except Exception as e:
        print(f"Error deleting document from Pinecone: {e}")
        return False