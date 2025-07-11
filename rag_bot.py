# rag_bot.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv
import os

# ----------------------
# Load environment variables
# ----------------------

load_dotenv()  # Load from .env file

# Azure OpenAI credentials from .env
os.environ["OPENAI_API_TYPE"] = os.getenv("OPENAI_API_TYPE")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")

# Azure Cognitive Search config from .env
index_name = "azure-rag-demo-index"
endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
key = os.getenv("AZURE_SEARCH_KEY")

# ----------------------
# Initialize services
# ----------------------

# Azure OpenAI LLM (chat model)
llm = AzureChatOpenAI(
    deployment_name="o4-mini",               # must match deployment name
    model="gpt-4-1106-preview",              # correct model for o4-mini
    temperature=1
)

# Azure Cognitive Search client
credential = AzureKeyCredential(key)
client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

# ----------------------
# Load and Process PDF
# ----------------------

def load_and_split_pdf(pdf_path="demo_paper.pdf"):
    print("Loading and splitting PDF...")
    loader = PyPDFLoader(pdf_path, extract_images=False)
    data = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.split_documents(data)
    return chunks

# ----------------------
# Upload to Azure Search
# ----------------------

def upload_to_search_index(chunks):
    print(f"Uploading {len(chunks)} chunks to Azure Search Index...")
    for index, chunk in enumerate(chunks):
        doc = {
            "id": str(index + 1),
            "data": chunk.page_content,
            "source": chunk.metadata["source"]
        }
        result = client.upload_documents(documents=[doc])
    print("Upload complete.")

# ----------------------
# RAG Retrieval Function
# ----------------------

def generate_response(user_question):
    print(f"Searching index for: {user_question}")
    context = ""
    results = client.search(search_text=user_question, top=2)

    for doc in results:
        context += "\n" + doc['data']

    qna_prompt_template = f"""
You will be provided with the question and a related context, you need to answer the question using the context.

Context:
{context}

Question:
{user_question}

Make sure to answer the question only using the context provided. If the context doesn't contain the answer, then return: "I don't have enough information to answer the question."

Answer:"""

    response = llm.invoke(qna_prompt_template)
    return response

# ----------------------
# Main Execution
# ----------------------

if __name__ == "__main__":
    chunks = load_and_split_pdf()
    upload_to_search_index(chunks)
    question = input("Ask a question based on the PDF: ")
    answer = generate_response(question)
    print("\nAnswer:", answer)
