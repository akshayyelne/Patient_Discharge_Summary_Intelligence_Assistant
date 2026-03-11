
---

# 🧠 Core AI Components

## 1. Structured Clinical Extraction

Discharge summaries are converted into structured JSON containing:

- Patient demographics
- Primary diagnosis
- Secondary diagnoses
- Procedures performed
- Medication changes
- Risk indicators
- Follow-up instructions

This step normalizes unstructured clinical documentation.

---

## 2. Retrieval Augmented Generation (RAG)

The system uses:

- SentenceTransformer embeddings
- FAISS vector search
- Context-aware LLM reasoning

This enables accurate question answering grounded in discharge content.

---

## 3. Multi-Patient Comparison Mode

Users can select multiple patients and compare:

- Length of stay
- Diagnoses
- Medication changes
- Risk indicators
- Discharge instructions

The system generates insights to support **discharge process optimization**.

---

# 🖥 User Interface

The project includes a **Gradio-based interactive interface** featuring:

- Multi-patient selector
- AI-powered chat interface
- Comparison mode
- Embedded project documentation

---

# 🛠 Technology Stack

- Python
- Google Colab
- Groq LLM (`openai/gpt-oss-120b`)
- SentenceTransformers
- FAISS
- Gradio
- PyPDF

---

# 📂 Running this Project on huggingface

https://huggingface.co/spaces/akshayyelne/discharge-intelligence-assistant

---

# 📊 Future Enhancements

Potential improvements include:

- Automated discharge KPI scoring
- Risk scoring models
- Visualization dashboards
- EMR system integration
- Real-time discharge monitoring

---

# 📄 Documentation

Detailed implementation documentation is available here:
docs/Patient_Discharge_Intelligence_Assistant_Project_Documentation.pdf


---

# 🎯 Use Case

This project demonstrates how hybrid **LLM + RAG architectures** can convert unstructured clinical documents into actionable insights that support hospital discharge optimization.

---

# 👨‍💻 Author

AI Portfolio Project exploring:

- Healthcare AI
- Retrieval Augmented Generation
- Clinical document intelligence

