import os
# Set the tokenizers parallelism before importing HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv 

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings




load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    streaming=True,
    api_key=GROQ_API_KEY
)
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


loader = TextLoader("yoda_galactic_feasts.txt")
documents = loader.load()



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

splits = text_splitter.split_documents(documents)


vectorstore = InMemoryVectorStore(embeddings_model)

vectorstore.add_documents(splits)


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

template = """
You are a helpful assistant that can answer questions about Yoda's Galactic Feast. 
You will be given a question and a context and you should base your answer on the context.
Question: {question}
Context: {context}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | llm | StrOutputParser()

question = "What is the address of Yoda's Galactic Feast?"

result = chain.invoke({"question": question, "context": retriever.invoke(question)})

message_history = []

def format_history():
    formatted = ""
    for msg in message_history:
        formatted += f"Human: {msg['question']}\nAssistant: {msg['answer']}\n\n"
    return formatted

print("\nWelcome to Yoda's Galactic Feast Chat! Type 'quit' to exit.\n")

while True:
    user_question = input("\nUser: ")
    if user_question.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    
    # Retrieve context from documents and extract text
    retrieved_docs = retriever.invoke(user_question)
    context_text = "\n".join(doc.page_content for doc in retrieved_docs)
    
    # Directly use the chain built earlier
    result = chain.invoke({
        "question": user_question,
        "context": context_text
    })
    
    print("\nAssistant:", result)

