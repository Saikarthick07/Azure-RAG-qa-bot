# Azure RAG Q&A Bot 🤖📄

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **Azure Cognitive Search**, **Azure OpenAI (GPT-4 Turbo)**, and **LangChain** to answer questions based on PDF documents.

---

## 📌 Features

- 🔍 Azure Cognitive Search-based indexing of PDF chunks
- 🧠 GPT-4 Turbo (`o4-mini`) for intelligent Q&A
- 📎 Uses LangChain and Python
- 🔐 API credentials securely managed via `.env` file
- 💻 MacBook-compatible, cross-platform friendly

---

## 🚀 Getting Started

### 1. **Clone the Repo**
```bash
git clone https://github.com/your-username/azure-rag-qa-bot.git
cd azure-rag-qa-bot
```
### 2. **Set Up Python Environment**

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

### 3. **Add Your .env File**

Create a .env file with the following variables:

```
# .env

# Azure OpenAI
OPENAI_API_TYPE=azure
OPENAI_API_VERSION=2024-12-01-preview
OPENAI_API_KEY=your-openai-api-key
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/

# Azure Cognitive Search
AZURE_SEARCH_ENDPOINT=https://your-search-resource.search.windows.net/
AZURE_SEARCH_KEY=your-search-admin-key
```
### 📄 **How It Works

- Loads and splits a PDF into chunks using LangChain
- Uploads the chunks to Azure Cognitive Search
- Accepts a user question
- Retrieves relevant chunks from the index
- Sends the question + context to GPT-4 via AzureChatOpenAI
- Prints a context-aware answer

### 🧪 Run the Bot

Ensure your .env and demo_paper.pdf are in the project root:

```
python rag_bot.py
```

You'll be prompted to ask a question, such as:
```
What are the challenges in deep learning for big data analytics?
```
Look for the RAG response from the model selected from the Azure OpenAI Fcatory.


