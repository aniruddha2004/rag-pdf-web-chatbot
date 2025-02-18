import os
import time
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Tavily Web Search
tavily_tool = TavilySearchResults()

# Embedding Model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Vector Database
persist_dir = "./chromadb_store"
vectorstore = Chroma(
    collection_name="sampleCollection",
    embedding_function=embedding_model,
    persist_directory=persist_dir
)

# Function to process PDF
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    vectorstore.add_documents(documents=chunks)

@app.route("/")
def upload_page():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Simulating file processing
    time.sleep(3)  # Show loading animation
    process_pdf(file_path)  # Store in vector database

    return jsonify({"success": True, "redirect": url_for("chat_page")})

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "")

    retriever = vectorstore.as_retriever()
    retrieval_tool = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    tools = [
        Tool(name="PDF_Retrieval", func=retrieval_tool.run, description="Retrieve answers from the uploaded PDF."),
        Tool(name="Web_Search", func=tavily_tool.invoke, description="Search the web for additional context.")
    ]

    # 🛠 FIX: Define the PromptTemplate properly
    prompt_template = PromptTemplate(
        input_variables=["input", "agent_scratchpad"],
        template="""
        You are a helpful assistant. Answer questions by using the tools provided.
        1. Check the uploaded PDF for the answer.
        2. If not found, use web search.

        Question: {input}

        Tools: {{tools}}

        {agent_scratchpad}
        """
    )

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": question})
    
    return jsonify({"answer": response["output"]})


if __name__ == "__main__":
    app.run(debug=True)
