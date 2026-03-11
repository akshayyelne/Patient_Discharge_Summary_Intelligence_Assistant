import os
import json
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
# Generate Structured Summary
# ==========================

def generate_structured_summary(patient_id):

    document_text = all_documents[patient_id]

    prompt = f"""
You are a clinical discharge extraction assistant.

Extract structured clinical information from the discharge summary.

Return JSON with EXACT structure:

{{
 "patient_demographics": {{
    "patient_name": "",
    "age": "",
    "gender": "",
    "length_of_stay": ""
 }},
 "primary_diagnosis": "",
 "secondary_diagnoses": [],
 "risk_flags": []
}}

Rules:
- Use ONLY this schema
- Do not invent new keys
- If information is missing return ""

Discharge Summary:
{document_text}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You extract structured clinical data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


# ==========================
# Process Uploaded Files
# ==========================

def process_files(files):

    global all_documents
    global all_chunks
    global chunk_metadata
    global all_embeddings
    global index
    global patient_summaries

    if not files:
        return []

    all_documents = {}
    all_chunks = []
    chunk_metadata = []
    patient_summaries = {}

    for file in files:

        document_text = extract_text(file)

        patient_id = file.name.split("/")[-1].replace(".pdf", "").replace(" ", "_")

        all_documents[patient_id] = document_text

        patient_chunks = chunk_text(document_text)

        for chunk in patient_chunks:
            all_chunks.append(chunk)
            chunk_metadata.append({"patient_id": patient_id})

    embeddings = embedding_model.encode(all_chunks)

    all_embeddings = np.array(embeddings).astype("float32")

    dimension = all_embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings)

    # Generate summaries
    for patient_id in all_documents.keys():
        patient_summaries[patient_id] = generate_structured_summary(patient_id)

    return list(all_documents.keys())


# ==========================
# Chat Function
# ==========================

def chat_with_memory(message, history, selected_patients):

    if history is None:
        history = []

    if not all_documents:
        history.append({"role": "assistant", "content": "⚠ Please upload discharge PDFs first."})
        return "", history

    if not selected_patients:
        history.append({"role": "assistant", "content": "⚠ Please select at least one patient."})
        return "", history

    if isinstance(selected_patients, str):
        selected_patients = [selected_patients]

    combined_summary = ""

    for patient in selected_patients:

        combined_summary += f"\n\n=== Patient: {patient} ===\n"
        combined_summary += str(patient_summaries.get(patient, ""))

    prompt = f"""
You are a clinical reasoning assistant.

Use ONLY the patient discharge summaries below.

{combined_summary}

Answer the following question clearly.

Question:
{message}
"""

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a clinical reasoning assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        answer = response.choices[0].message.content

    except Exception as e:

        answer = f"⚠ Model Error: {str(e)}"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return "", history


# ==========================
# Dashboard Generator
# ==========================

def generate_dashboard(selected_patients):

    if not selected_patients:
        return "## 🧠 Clinical Comparison Dashboard\nSelect patients to view insights."

    if isinstance(selected_patients, str):
        selected_patients = [selected_patients]

    table = "| Patient File | Name | Age | Diagnosis | Length of Stay | Risk Flags |\n"
    table += "|--------------|------|-----|-----------|---------------|-----------|\n"

    for patient in selected_patients:

        summary_json = patient_summaries.get(patient, "{}")

        try:

            data = json.loads(summary_json)

            demographics = data.get("patient_demographics", {})

            name = (
                demographics.get("patient_name")
                or demographics.get("name")
                or demographics.get("patient")
                or "N/A"
            )

            age = demographics.get("age", "N/A")
            diagnosis = data.get("primary_diagnosis", "N/A")
            los = demographics.get("length_of_stay", "N/A")

            risk_flags = data.get("risk_flags", [])

            if isinstance(risk_flags, list):
                risk_flags = ", ".join(risk_flags)

        except Exception:

            name = "N/A"
            age = "N/A"
            diagnosis = "N/A"
            los = "N/A"
            risk_flags = "N/A"

        table += f"| {patient} | {name} | {age} | {diagnosis} | {los} | {risk_flags} |\n"

    return "## 🧠 Clinical Comparison Dashboard\n\n" + table


# ==========================
# Gradio UI
# ==========================

with gr.Blocks() as demo:

    gr.Markdown("# 🏥 Patient Discharge Intelligence Assistant")

    gr.Markdown("Upload discharge summaries and analyse patient discharge outcomes.")

    with gr.Row():

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

    dashboard = gr.Markdown(
        "## 🧠 Clinical Comparison Dashboard\nSelect patients to view insights."
    )

    chatbot = gr.Chatbot(height=400)

    msg = gr.Textbox(
        label="Ask a clinical question",
        placeholder="Example: Compare readmission risks between patients."
    )

    # Upload Handler

    def handle_upload(files):

        patients = process_files(files)

        return gr.Dropdown(choices=patients, value=[])

    file_upload.upload(
        handle_upload,
        inputs=file_upload,
        outputs=patient_dropdown
    )

    # Dashboard Update

    patient_dropdown.change(
        generate_dashboard,
        inputs=patient_dropdown,
        outputs=dashboard
    )

    # Chat Interaction

    msg.submit(
        chat_with_memory,
        inputs=[msg, chatbot, patient_dropdown],
        outputs=[msg, chatbot]
    )


demo.queue().launch()
