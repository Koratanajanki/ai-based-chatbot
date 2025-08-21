from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------
# Step1: Load PDF
# ---------------------------
loader = PyPDFLoader(r"E:\newchatbot\tcschatbot\documents\tcsreport1.pdf")
documents = loader.load()
print("✅ Total pages loaded:", len(documents))

# ---------------------------
# Step2: Split into chunks
# ---------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=500,
)
chunks = text_splitter.split_documents(documents)
print("✅ Total chunks created:", len(chunks))

# ---------------------------
# Step3: Create embeddings
# ---------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/msmarco-MiniLM-L-6-v3"
)

# ---------------------------
# Step4: Build FAISS vector DB
# ---------------------------
vector_store = FAISS.from_documents(chunks, embeddings)
print("✅ Successfully created FAISS vector database")

# ---------------------------
# Step5: Save FAISS locally
# ---------------------------
vector_store.save_local("tcs_doc_index")
print("🎉 Successfully stored FAISS vector DB in 'tcs_doc_index'")
