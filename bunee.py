from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from pathlib import Path

# 1. HuggingFace pipeline oluştur (örneğin, flan-t5-small)
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_length=256,
    do_sample=True,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=generator)

# 2. Embed model
embed_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Qdrant bağlantısı
qdrant = Qdrant(
    url="http://localhost:6333",
    collection_name="wiki_collection",
    embeddings=embed_model,
    prefer_grpc=False
)

# 4. Load documents
def load_documents(data_dir):
    docs = []
    for file in Path(data_dir).glob("*.json"):
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
            docs.append({"page_content": data["content"], "metadata": {"title": data["title"]}})
    return docs

# 5. Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

def split_documents(docs):
    split_docs = []
    for doc in docs:
        chunks = splitter.split_text(doc["page_content"])
        for chunk in chunks:
            split_docs.append({"page_content": chunk, "metadata": doc["metadata"]})
    return split_docs

# 6. Vektör veritabanına ekle
def ingest_documents(docs):
    qdrant.add_documents(docs)

# 7. Sorgu + cevap zinciri
def run_qa():
    retriever = qdrant.as_retriever(search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    while True:
        query = input("Soru (q ile çık): ")
        if query.lower() == "q":
            break
        answer = qa.run(query)
        print("\nCevap:", answer)

if __name__ == "__main__":
    DATA_DIR = "data"  # JSON dosyalarının olduğu klasör
    docs = load_documents(DATA_DIR)
    split_docs = split_documents(docs)
    ingest_documents(split_docs)
    run_qa()
