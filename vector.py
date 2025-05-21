import os
import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CSV_PATH = "medquad_clean.csv"
CHROMA_PATH = "./chroma_langchain_db"
EMBED_MODEL = "nomic-embed-text:v1.5"

add_documents = not os.path.exists(CHROMA_PATH)

# 🔹 Sadece 500 satır al
df = pd.read_csv(CSV_PATH).head(500)

embedding_model = OllamaEmbeddings(model=EMBED_MODEL)



if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        doc = Document(page_content=answer, metadata={"question": question})
        documents.append(doc)
        ids.append(str(i))

vector_store = Chroma(
    collection_name="medical_qa_sheet",
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)

if add_documents:
    vector_store.add_documents(documents=documents,ids=ids)


retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
    )
