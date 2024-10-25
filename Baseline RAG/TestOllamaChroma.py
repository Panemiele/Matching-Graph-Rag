from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
import os
import dotenv


dotenv.load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
model_local = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192")

# 1. Split data into chunks
urls = [
    "https://ollama.com/",
    "https://ollama.com/blog/windows-preview",
    "https://ollama.com/blog/openai-compatibility",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list)

# 2. Convert documents to Embeddings and store them
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text', show_progress=True),
    persist_directory="./local-rag"
)
retriever = vectorstore.as_retriever()

# 3. Before RAG (Answer is not good, look below for an example)
#
# Ollama!
# Ollama is a unique and fascinating concept. It's a game that originated in Colombia, South America, and has gained popularity worldwide.
# In Ollama, players take turns tossing a small ball (called an "ollama") to each other while simultaneously performing a specific action, usually a dance move or a silly gesture. The actions can be anything from simple movements like spinning around or making funny faces to more complex moves like doing a quick dance routine or even playing a musical instrument!
# The objective is not only to catch the ball but also to complete the assigned task without dropping the ball or messing up the action. If you succeed, you earn points and get to keep the game going! If you fail, the next player gets a chance to try.
# Ollama has become an integral part of Colombian culture, with many people playing it as a social activity, often at parties, festivals, or just for fun. It's also gained popularity on social media platforms like TikTok and YouTube, where people share their Ollama moments and compete with each other.
# Would you like to learn more about the rules of Ollama or perhaps even try playing it?
print("Before RAG\n")
before_rag_template = "What is {topic}"
before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
before_rag_chain = before_rag_prompt | model_local | StrOutputParser()
print(before_rag_chain.invoke({"topic": "Ollama"}))



# 4. After RAG (Now the answer is clean and correct, check below)
#
# Based on the provided context, Ollama appears to be a company or organization that provides an API for building conversational AI applications. The company seems to offer various models, including "Llama", "Phi", "Mistral", and "Gemma 2", which can be used to create custom language models.
print("\n########\nAfter RAG\n")
after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)
print(after_rag_chain.invoke("What is Ollama?"))

# loader = PyPDFLoader("Ollama.pdf")
# doc_splits = loader.load_and_split()
