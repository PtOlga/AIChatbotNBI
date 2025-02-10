import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.6,
    api_key=GROQ_API_KEY
)

embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

loader = TextLoader("yoda_galactic_feasts.txt")

document = loader.load()

chunks = text_splitter.split_documents(document)

vector_store = InMemoryVectorStore.from_documents(chunks, embeddings_model)

retriever = vector_store.as_retriever()

template = """
You are a helpful assistant that can answer questions about the Yoda's Galactic Feast restaurant.

Answer in the tone of Yoda from Star Wars.
Question: {question}

Previous conversation:
{chat_history}

Only use the provided context to answer the question.

Context: {context}
"""

prompt = PromptTemplate.from_template(template)

def format_history(message_history):
    formatted = ""
    for msg in message_history:
        formatted += f"Human: {msg['question']}\nAssistant: {msg['answer']}\n\n"
    return formatted

chain = {
    "context": retriever,
    "question": RunnablePassthrough(),
    "chat_history": RunnableLambda(lambda x: format_history(message_history))
} | prompt | llm | StrOutputParser()

message_history = []

print("\nWelcome to Yoda's Galactic Feast Chat! Type 'quit' to exit.\n")

while True:
    user_question = input("\nUser: ")
    if user_question.lower() in ["quit", "exit", "q"]:
        print("\nMay the Force be with you, farewell!")
        break
    
    result = chain.invoke(user_question)
    
    # Store the interaction in message history
    message_history.append({
        "question": user_question,
        "answer": result
    })
    
    print("\nAssistant:", result)