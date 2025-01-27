# PDF & Web-Based Chatbot with OpenAI and Tavily

## Overview
This project demonstrates an **agentic Retrieval-Augmented Generation (RAG)** implementation. The chatbot combines the power of retrieval-based question answering and real-time web searches. It enables users to:

- Extract answers from uploaded PDFs through a vector-based retrieval system.
- Retrieve additional context via web searches using Tavily when the answer isn't found in the PDF.
- Leverage a tool-calling agent that orchestrates interactions between the tools and the OpenAI language model.

Built with OpenAI, LangChain, Tavily, Chroma, and Gradio, this project offers a seamless and intelligent way to interact with document content and online information.


---

## Features
- **PDF-Based Question Answering**: Extracts and retrieves answers from uploaded PDF documents.
- **Web Search Integration**: Performs web searches for questions outside the PDF scope using Tavily.
- **Interactive UI**: Built with Gradio for a user-friendly experience.

---

## Tech Stack
- **Frontend**: Gradio for the chatbot interface.
- **Backend**:
  - LangChain for chaining LLM capabilities.
  - OpenAI's GPT-4o-mini model for LLM responses.
  - Tavily for web search integration.
  - Chroma for vector database storage.
- **Environment Management**: dotenv for managing environment variables.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
   
  2. **Set up a virtual environment**:
     ```bash
     python -m venv myenv
     source myenv/bin/activate  # On Windows: myenv\Scripts\activate
     ```
     
  3. **Install dependencies**:
     ```bash
     pip install -r requirements.txt
     ```
     
  4. **Set up .env file**:Create a .env file based on .env.example and provide your OpenAI API key and other necessary configurations.
 
  5. **Run the application**:
     ```bash
     python src/main.py
     ```
     

---

## Usage
1. **Launch the Application**: Start the chatbot application and access the interface.
2. **Upload a PDF**: Use the file upload option to add a PDF document. The application will process and index the document for easy retrieval.
3. **Ask Questions**: Enter your questions in the provided text box. The chatbot will first search for relevant answers within the uploaded PDF. If the answer is not found, it will perform a web search to retrieve additional information.
4. **Receive Answers**: The chatbot will provide concise and relevant responses based on the content of the PDF or information from the web.

---

## Directory Structure
The project is organized to ensure a clean separation of functionality:
- **PDF storage and processing** are managed in a dedicated directory.
- **Source code** is located within a structured folder to ensure modularity.
- **Environment variables** are used for secure and configurable API key storage.

---

## Contributing
Feedback and contributions are always welcome! Users can report issues, suggest features, or contribute code to improve the project further.

---

## Acknowledgments
Special thanks to:
- **OpenAI** for powering the chatbot with GPT models.
- **LangChain** for enabling retrieval and tool-based agents.
- **Tavily** for web search functionality.
- **Gradio** for the interactive user interface.
- **Chroma** for managing vector databases.

---


  