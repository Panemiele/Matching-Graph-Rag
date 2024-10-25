from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

local_path = "WEF_The_Global_Cooperation_Barometer_2024.pdf"
# if local_path:
#     loader = UnstructuredPDFLoader(file_path=local_path, mode="elements", strategy="fast")
#     print("Loading...")
#     data = loader.load()
#     print("Done")
# else:
#     print("Non sei nell'if")
loader = PyPDFLoader(local_path)
data = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunks = text_splitter.split_documents(data)



# import chromadb
# chroma_client = chromadb.HttpClient(host="localhost", port=8000)
# print(chroma_client.list_collections())
# chroma_client.delete_collection(name="local-rag")
# collection = chroma_client.create_collection(name="local-rag")

ollama_embeddings = OllamaEmbeddings(model="llama3", show_progress=True)

vector_db = Chroma.from_documents(
    documents=chunks[:10],
    embedding=ollama_embeddings,
    collection_name="local-rag"
)
print("ok")


local_model = "llama3"
llm = ChatOllama(model=local_model)
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# chain.invoke(input(""))


chain.invoke("What are the 5 pillars of global cooperation?")


# Delete all collections in the db
vector_db.delete_collection()


