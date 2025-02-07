import os
# Set the tokenizers parallelism before importing HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
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


loader = WebBaseLoader("https://en.wikipedia.org/wiki/Black_hole")
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
You are a helpful assistant that can answer questions about Black holes. 
You will be given a question and a context and you should base your answer on the context.
Question: {question}
Context: {context}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | llm | StrOutputParser()

question = "What is a black hole?"

result = chain.invoke({"question": question, "context": retriever.invoke(question)})

message_history = []

def format_history():
    formatted = ""
    for msg in message_history:
        formatted += f"Human: {msg['question']}\nAssistant: {msg['answer']}\n\n"
    return formatted

print("\nWelcome to the Black Hole Chat! Type 'quit' to exit.\n")

while True:
    user_question = input("\nUser: ")
    
    if user_question.lower() == 'quit':
        print("\nGoodbye!")
        break
    
    # Get context and generate response
    context = retriever.invoke(user_question)
    
    # Update template to include chat history
    template_with_history = """
    You are a helpful assistant that can answer questions about Black holes. 
    You will be given a question and a context and you should base your answer on the context. 

    Take the chat history into account when answering the question.
    Chat history:
    {history}

    Question: {question}

    Context: {context}

    Answer:
    """
    
    prompt_with_history = ChatPromptTemplate.from_template(template_with_history)
    chain_with_history = prompt_with_history | llm | StrOutputParser()
    
    # Generate response with history
    result = chain_with_history.invoke({
        "question": user_question,
        "context": context,
        "history": format_history()
    })
    
    # Store the interaction
    message_history.append({
        "question": user_question,
        "answer": result
    })
    
    # Print the response
    print("\nAssistant:", result)

