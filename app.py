from flask import Flask, jsonify, request, render_template
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

# ---------------------------
# Load FAISS + Embeddings (only once at startup)
# ---------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-MiniLM-L-6-v3")
new_vector_store = FAISS.load_local(
    "tcs_doc_index", embeddings, allow_dangerous_deserialization=True
)
print("✅ FAISS index loaded successfully")

# ---------------------------
# Routes
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/tcs", methods=["POST"])
def tcs_chatbot_api():
    data = request.get_json()
    question = data.get("tcs_question", "")

    try:
        # 🔍 Search for relevant context
        context_docs = new_vector_store.similarity_search(question, k=2)
        context = "\n".join([doc.page_content for doc in context_docs])

        prompt = f"""
        Answer the question using only the given context from TCS Annual Reports 2024 and 2025.
        Also handle casual questions like CEO, company locations, hiring numbers per year, and work environment.
        Do NOT answer if the question is outside company-related topics or unknown.
        Give short answers (2–3 lines), or max 5 lines if needed.

        Context: {context}
        Question: {question}
        """

        # 🤖 Call OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for TCS Annual Reports."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content.strip()
        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
