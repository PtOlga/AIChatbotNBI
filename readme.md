# Yoda's Galactic Feast Chatbot 🤖🍽️

Welcome, young Padawan! This project demonstrates how to build an interactive chatbot using Python and LangChain. The chatbot serves as a virtual assistant for Yoda's Galactic Feast restaurant, responding in Master Yoda's iconic speaking style.

## 🎯 Learning Objectives

Through this project, you'll learn about:
- Building conversational AI applications
- Working with Large Language Models (LLMs)
- Using vector stores for semantic search
- Implementing chat history
- Managing environment variables
- Creating interactive command-line interfaces

## 🛠️ Technologies Used

- **Python**: The primary programming language
- **LangChain**: Framework for building LLM applications
- **Groq**: LLM provider for natural language processing
- **HuggingFace**: For text embeddings
- **python-dotenv**: For environment variable management
- **LangSmith**: For monitoring and debugging LLM applications

## 📋 Prerequisites

- Python 3.10 or higher
- Basic understanding of Python programming
- A Groq API key (sign up at https://groq.com)

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <repository-url>
cd AIchatbotNBI
```

### 2. Create a Virtual Environment
Open the terminal in your IDE and run the following commands:

```bash
# Create virtual environment
python -m venv venv

# Activate the virtual environment
# For Linux/MacOS:
source venv/bin/activate

# For Windows (CMD):
.\venv\Scripts\activate.bat

# For Windows (PowerShell):
.\venv\Scripts\activate.ps1

# For Windows with Git Bash:
source venv/Scripts/activate
```

For VS Code users: Press `Ctrl+Shift+P` -> Select "Python: Select Interpreter" -> Choose the `venv/bin/python` interpreter.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root and add your API keys:
```
GROQ_API_KEY="your_groq_api_key_here"
LANGCHAIN_API_KEY="your_langsmith_api_key_here"
LANGCHAIN_TRACING_V2=true
LANGSMITH_ENDPOINT="https://eu.api.smith.langchain.com"
LANGSMITH_PROJECT="yoda-galactic-feast"
```

### 5. Prepare Your Data
The chatbot uses a text file (`yoda_galactic_feasts.txt`) containing information about the restaurant. You can modify this file to customize the bot's knowledge base.

### 6. Run the Chatbot
```bash
python main.py
```

## 🧠 How It Works

### Main Components

1. **Document Loading and Splitting**
```python
loader = TextLoader("yoda_galactic_feasts.txt")
document = loader.load()
chunks = text_splitter.split_documents(document)
```
This code loads the restaurant's information and splits it into manageable chunks.

2. **Vector Store Creation**
```python
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings_model)
retriever = vector_store.as_retriever()
```
The text chunks are converted into vectors for semantic search.

3. **Chat History Management**
```python
def format_history(message_history):
    formatted = ""
    for msg in message_history:
        formatted += f"Human: {msg['question']}\nAssistant: {msg['answer']}\n\n"
    return formatted
```
This function maintains the conversation context.

4. **Chain Configuration**
```python
chain = {
    "context": retriever,
    "question": RunnablePassthrough(),
    "chat_history": RunnableLambda(lambda x: format_history(message_history))
} | prompt | llm | StrOutputParser()
```
This sets up the processing pipeline for handling user queries.

## 🔍 Key Concepts

1. **Vector Stores**: Used for semantic search to find relevant information based on user queries.
2. **Embeddings**: Convert text into numerical vectors that capture semantic meaning.
3. **Prompt Engineering**: Structuring the input to get desired responses from the LLM.
4. **Chain**: A sequence of operations that process user input and generate responses.

## 🎮 Using the Chatbot

1. Start the chatbot using `python main.py`
2. Type your questions about the restaurant
3. The bot will respond in Yoda's style using information from its knowledge base
4. Type 'quit', 'exit', or 'q' to end the conversation

## 🛠️ Customization Options

1. Modify the `template` string to change the bot's personality
2. Adjust the `chunk_size` and `chunk_overlap` in `text_splitter` for different document processing
3. Change the `temperature` parameter in the LLM configuration to control response creativity

## 📚 Learning Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Vector Store Concepts](https://python.langchain.com/docs/how_to/#vector-stores)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

## 🤝 Contributing

Feel free to fork this project and customize it for your own learning! Some ideas for extensions:
- Add support for multiple restaurants
- Implement a web interface
- Add more sophisticated conversation handling
- Integrate with a database for persistent storage

## ⚠️ Common Issues and Solutions

1. **ModuleNotFoundError**: Make sure you've activated your virtual environment and installed all requirements
2. **API Key Error**: Check that your `.env` file is properly configured
3. **Memory Issues**: Reduce chunk size if processing large documents

## 📊 Monitoring and Debugging

This project uses LangSmith for monitoring and debugging the LLM application. LangSmith provides:

1. **Trace Visualization**: See how your chains and agents process requests
2. **Performance Monitoring**: Track latency, token usage, and costs
3. **Debug Interface**: Inspect intermediate steps and outputs
4. **Dataset Management**: Create and manage evaluation datasets

To access these features:
1. Sign up for LangSmith at [smith.langchain.com](https://smith.langchain.com/)
2. Get your API key from the LangSmith dashboard
3. Add the key and other LangSmith configurations to your `.env` file
4. Visit the LangSmith dashboard to monitor your application
