import os
from markdownify import markdownify as md_to_text
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
import gradio as gr
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import warnings

load_dotenv()

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize Tavily for Web Search
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

# Function to process PDF and store embeddings
def process_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    vectorstore.add_documents(documents=chunks)
    return "PDF processed and added to the vector database!"

# Retrieval-based QA Chain
retriever = vectorstore.as_retriever()
retrieval_tool = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Define Tools
pdf_retrieval_tool = Tool(
    name="PDF_Retrieval",
    func=retrieval_tool.run,
    description="Retrieve answers from the uploaded PDF."
)

tavily_search_tool = Tool(
    name="Web_Search",
    func=tavily_tool.invoke,
    description="Search the web for additional context."
)

tools = [pdf_retrieval_tool, tavily_search_tool]

# Prompt Template
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""
    You are a helpful assistant. Answer questions by using the tools provided. 
    1. First, check the uploaded PDF for the answer.
    2. If not found, use web search for additional information.

    Question: {input}

    Tools: {{tools}}

    Thought: Let's think step by step.
    {agent_scratchpad}
    """
)

# Create the agent
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

# Create AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Gradio Interface Functions
def ask_question(question):
    try:
        response = agent_executor.invoke({"input": question})
        plain_text = md_to_text()  # Convert Markdown to plain text
        print(plain_text)
        return plain_text
    except Exception as e:
        return f"An error occurred: {str(e)}"

def upload_pdf(file):
    return process_pdf(file)

# Gradio UI
with gr.Blocks(css="""
.gradio-container {
    max-width: 800px; 
    margin: auto; 
    padding: 20px;
}
.file-upload {
    height: 150px !important; /* Set consistent height */
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid #ddd;
    border-radius: 8px;
    font-family: Arial, sans-serif;
    font-size: 14px;
}
""") as chatbot_ui:
    gr.Markdown(
        """
        <h1 style="text-align: center; font-family: Arial, sans-serif; color: #2C3E50;">
            PDF & Web-Based Chatbot with OpenAI and Tavily
        </h1>
        <p style="text-align: center; font-family: Arial, sans-serif; color: #7F8C8D;">
            Upload a PDF and ask questions based on its content or perform web-based searches.
        </p>
        """,
    )
    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            pdf_input = gr.File(
                label="Upload PDF", 
                file_types=[".pdf"], 
                interactive=True, 
                type="filepath",  # File path type
                elem_classes=["file-upload"]  # Add custom CSS class
            )
        with gr.Column(scale=1):
            pdf_submit = gr.Button("Process PDF", variant="primary")
        pdf_status = gr.Textbox(label="Status", interactive=False, placeholder="Waiting for PDF upload...")

    gr.Markdown("<hr style='border:1px solid #ddd;' />")  # Horizontal Divider
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Ask a Question", 
                placeholder="Type your question here...", 
                lines=2
            )
        with gr.Column(scale=1):
            question_submit = gr.Button("Get Answer", variant="primary")
    answer_output = gr.Textbox(
        label="Answer", 
        interactive=False, 
        placeholder="The answer will appear here...", 
        lines=6
    )

    # Define Actions
    pdf_submit.click(upload_pdf, inputs=[pdf_input], outputs=[pdf_status])
    question_submit.click(ask_question, inputs=[question_input], outputs=[answer_output])

chatbot_ui.launch()
