from typing import Annotated, List, Dict, Tuple, TypedDict
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import InMemoryVectorStore
from operator import itemgetter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define the state
class State(TypedDict):
    messages: List[Dict]
    context: List[Dict]
    next: str

# Setup your tools and models as before
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    streaming=True
)

# Add this function to clean the text
def clean_wiki_text(text: str) -> str:
    # Remove multiple newlines and whitespace
    text = re.sub(r'\n\s*\n', '\n', text)
    # Remove navigation elements and other UI text
    text = re.sub(r'Jump to content|Main menu|Navigation|Search|Appearance|Personal tools|Contents|move to sidebar|hide', '', text)
    # Remove empty lines
    text = '\n'.join(line for line in text.splitlines() if line.strip())
    # Remove references like [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    return text

# Modify the document loading part
loader = WebBaseLoader(
    "https://en.wikipedia.org/wiki/Black_hole",
    bs_kwargs={
        "parse_only": BeautifulSoup.SoupStrainer(["p", "h1", "h2", "h3"]),  # Only get main content
    }
)
documents = loader.load()

# Clean the documents
for doc in documents:
    doc.page_content = clean_wiki_text(doc.page_content)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
splits = text_splitter.split_documents(documents)
vectorstore = InMemoryVectorStore(embeddings_model)
vectorstore.add_documents(splits)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Define the functions
def retrieve(state: State) -> State:
    # Get the last user message
    last_message = state["messages"][-1].content
    # Retrieve relevant documents
    state["context"] = retriever.invoke(last_message)
    return state

def generate_answer(state: State) -> State:
    # Convert Document objects to string context
    context = "\n".join(doc.page_content for doc in state["context"])
    
    # Create a proper prompt
    prompt = f"""
    You are a helpful assistant that can answer questions about Black holes. 
    Answer the question based on the following context.
    
    Question: {state["messages"][-1].content}
    Context: {context}
    
    Answer:
    """
    
    # Get response
    response = llm.invoke(prompt)
    state["messages"].append(AIMessage(content=response.content))
    return state


# Create the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("retriever", retrieve)
workflow.add_node("llm", generate_answer)

# Add edges
workflow.add_edge(START, "retriever")
workflow.add_edge("retriever", "llm")
workflow.add_edge("llm", END)

# Compile the graph
chain = workflow.compile()

def get_chat_response(message: str):
    result = chain.invoke({
        "messages": [HumanMessage(content=message)],
        "context": [],
        "next": ""
    })
    return result["messages"][-1].content

for step in chain.stream({"messages": [HumanMessage(content="What is a black hole?")]}, stream_mode="updates"):
    print(f"{step}\n\n----------------\n")