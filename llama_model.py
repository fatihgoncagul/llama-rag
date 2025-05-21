from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # retriever, vector.py'deki db.as_retriever(...)

# Ollama modelini başlat
model = OllamaLLM(model="llama3.1")

# Prompt şablonu
template = """
You are a health assistant who answers questions and makes suggestions.

Here are some relevant reviews:
{reviews}

Here is the question to answer:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-----------------------")
    question = input("Ask your question (q to quit): ").strip()
    if question.lower() == "q":
        break

    documents = retriever.invoke(question)

    reviews_text = "\n\n".join([
        f"Q: {doc.metadata.get('question', '')}\nA: {doc.page_content}"
        for doc in documents
    ])

    result = chain.invoke({
        "reviews": reviews_text,
        "question": question
    })

    print("\n\n--- Answer ---\n")
    print(str(result))
