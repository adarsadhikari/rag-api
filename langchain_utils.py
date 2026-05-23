from dotenv import load_dotenv
import os
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import List
from langchain_core.documents import Document
import os
from pincone_utils import vectorstore


retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
output_parser = StrOutputParser()

contextualize_q_system_prompt=(
    "Given a chat history and latest user question "
    "which might reference context in chat history, "
    "formulate a standalone question that can be understood "
    "without the chat history. DO NOT answer the question, "
    "just reformulate it if needed else return the question as it is."
)

contextualize_q_prompt=ChatPromptTemplate.from_messages([
    ("system",contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
])

qa_prompts=ChatPromptTemplate.from_messages([
    ('system','you are a helpful expert assistant. Use the following context to answer the questions'),
    ('system','Context:{context}'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{input}')
])

def get_rewritten_question(inputs):
    return {
        "input": inputs["input"],
        "chat_history": inputs["chat_history"],
    }

# llm_model = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-V3.2",
#     task="conversational",
#     max_new_tokens=1024,
#     temperature=0.7,
# )

MODEL_MAP = {
    "GPT-oss-120b": "openai/gpt-oss-120b",
    "GPT-oss-20b": "openai/gpt-oss-20b",
    "DeepSeek-V3.2": "deepseek-ai/DeepSeek-V3.2"
}

def get_rag_chain(model: str):
    repo=MODEL_MAP.get(model)
    base_llm = HuggingFaceEndpoint(
    repo_id=repo,
    task="conversational",
    max_new_tokens=1024,
    temperature=0.7,
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
    )
    llm = ChatHuggingFace(llm=base_llm)
    contextualize_chain = (
        contextualize_q_prompt
        | llm
        | StrOutputParser()
    )

    retrieval_chain = (
        contextualize_chain
        | retriever
    )

    rag_chain = (
        {
            "context": retrieval_chain,
            "input": RunnableLambda(lambda x: x["input"]),
            "chat_history": RunnableLambda(lambda x: x["chat_history"]),
        }
        | qa_prompts
        | llm
        | StrOutputParser()
        |  RunnableLambda(lambda x: {"answer": x})
    )

    return rag_chain

