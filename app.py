import os
import gradio as gr
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# ==========================
# Initialize Models
# ==========================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)


# ==========================
# Global Storage
# ==========================

all_documents = {}
all_chunks = []
chunk_metadata = []
all_embeddings = None
index = None
patient_summaries = {}


# ==========================
# Extract Text from PDF
# ==========================

def extract_text(file):

    reader = PdfReader(file.name)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# ==========================
# Chunking
# ==========================

def chunk_text(text, chunk_size=800, overlap=150):

    chunks = []

    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])

    return chunks


# ==========================
# Process Uploaded Files
# ==========================

def process_files(files):

    global all_documents
    global all_chunks
    global chunk_metadata
    global all_embeddings
    global index

    all_documents = {}
    all_chunks = []
    chunk_metadata = []

    for file in files:

        document_text = extract_text(file)

        patient_id = file.name.split("/")[-1].replace(".pdf", "").replace(" ", "_")

        all_documents[patient_id] = document_text

        patient_chunks = chunk_text(document_text)

        for chunk in patient_chunks:

            all_chunks.append(chunk)

            chunk_metadata.append({
                "patient_id": patient_id
            })

    # Generate embeddings
    embeddings = embedding_model.encode(all_chunks)
    all_embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    dimension = all_embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings)

    return list(all_documents.keys())


# ==========================
# Patient Retrieval
# ==========================

def retrieve_patient_context(query, selected_patient, k=5):

    query_embedding = embedding_model.encode([query]).astype("float32")

    patient_indices = [
        i for i, meta in enumerate(chunk_metadata)
        if meta["patient_id"] == selected_patient
    ]

    if not patient_indices:
        return []

    patient_embeddings = all_embeddings[patient_indices]

    temp_index = faiss.IndexFlatL2(all_embeddings.shape[1])
    temp_index.add(patient_embeddings)

    distances, indices = temp_index.search(
        query_embedding,
        min(k, len(patient_indices))
    )

    retrieved_chunks = []

    for idx in indices[0]:
        retrieved_chunks.append(all_chunks[patient_indices[idx]])

    return retrieved_chunks


# ==========================
# Generate Structured Summary
# ==========================

def generate_structured_summary(patient_id):

    document_text = all_documents[patient_id]

    prompt = f"""
You are a clinical discharge extraction assistant.

Extract structured clinical information from the content below.

Return STRICT JSON.

Content:
{document_text}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


# ==========================
# Chat Function
# ==========================

def chat_with_memory(message, history, selected_patients):

    if not selected_patients:
        history.append((message, "⚠ Please select at least one patient."))
        return "", history

    if isinstance(selected_patients, str):
        selected_patients = [selected_patients]

    combined_summary = ""

    for patient in selected_patients:

        if patient not in patient_summaries:
            patient_summaries[patient] = generate_structured_summary(patient)

        combined_summary += f"\n\n=== Patient: {patient} ===\n"
        combined_summary += str(patient_summaries[patient])

    # Convert Gradio history → Groq message format
    messages = [
        {"role": "system", "content": "You are a clinical reasoning assistant."}
    ]

    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    # Add the new user message
    messages.append({
        "role": "user",
        "content": f"""
Use ONLY the patient data below.

{combined_summary}

Question:
{message}
"""
    })

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.2
    )

    answer = response.choices[0].message.content

    history.append((message, answer))

    return "", history


# ==========================
# Gradio UI
# ==========================

with gr.Blocks() as demo:

    gr.Markdown("# 🏥 Patient Discharge Intelligence Assistant")

    file_upload = gr.File(
        label="Upload Discharge PDFs",
        file_types=[".pdf"],
        file_count="multiple"
    )

    patient_dropdown = gr.Dropdown(
        choices=[],
        label="Select Patient(s)",
        multiselect=True
    )

    chatbot = gr.Chatbot(height=500)

    msg = gr.Textbox(label="Ask a question")

    def handle_upload(files):

        patients = process_files(files)

        return gr.Dropdown(choices=patients, value=[])

    file_upload.upload(
        handle_upload,
        inputs=file_upload,
        outputs=patient_dropdown
    )

    msg.submit(
        chat_with_memory,
        inputs=[msg, chatbot, patient_dropdown],
        outputs=[msg, chatbot]
    )


demo.launch()
