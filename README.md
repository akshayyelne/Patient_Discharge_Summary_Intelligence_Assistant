**🏥 Patient Discharge Intelligence Assistant**

An AI-powered clinical document intelligence system that analyzes hospital discharge summaries and enables multi-patient comparison, risk identification, and clinical reasoning using Retrieval-Augmented Generation (RAG).

This project demonstrates how Large Language Models + vector search can transform unstructured clinical documentation into structured insights that support hospital discharge optimisation and patient safety.

**🚀 Live Application**

Run the application here:

👉 https://huggingface.co/spaces/akshayyelne/discharge-intelligence-assistant

The application allows users to:

   Upload discharge summary PDFs

   Extract structured clinical information

  Compare multiple patients

  Ask clinical reasoning questions grounded in discharge data


**🧠 Core AI Components**

1️⃣ Structured Clinical Extraction

   Discharge summaries are converted into structured JSON containing:

   Patient demographics

   Primary diagnosis

   Secondary diagnoses

   Procedures performed

   Medication changes

   Risk indicators

   Follow-up instructions

   This step normalizes unstructured clinical documentation into machine-readable clinical data.
**
2️⃣ Retrieval-Augmented Generation (RAG)**

The system uses:

   SentenceTransformer embeddings

   FAISS vector similarity search

   Context-aware LLM reasoning

This architecture ensures AI responses are grounded in the discharge summaries, reducing hallucination and improving reliability.

**3️⃣ Multi-Patient Clinical Comparison**

   Users can select multiple patients and compare:

   Length of stay

   Primary diagnosis

   Medication changes

   Risk indicators

   Discharge instructions

The system generates insights that can support discharge process optimisation and clinical risk analysis.

**📊 Clinical Comparison Dashboard**

The system automatically generates a clinical comparison dashboard showing:

  Patient name

  Age

  Diagnosis

  Length of stay

  Risk indicators

This provides a quick overview of discharge outcomes across multiple patients.

**
💬 AI Clinical Assistant**

Users can ask clinical questions such as:

 Which patient has the highest discharge risk?

 Compare diagnoses between selected patients

 Identify potential discharge complications

 Summarize medication changes

 The AI assistant answers using only the uploaded discharge summaries.

**🖥 User Interface**

The project includes an interactive Gradio interface featuring:

 Multi-patient document upload

 Patient selection

 Clinical comparison dashboard

 AI-powered chat assistant

 Structured clinical data extraction

**🛠 Technology Stack**

 Component	Technology
 Language	Python
 Development	Google Colab
 LLM	Groq (openai/gpt-oss-120b)
 Embeddings	SentenceTransformers
 Vector Database	FAISS
 UI	Gradio
 PDF Processing	PyPDF
 Running the Project
 Run via Hugging Face

**The easiest way to run the application:**

👉 https://huggingface.co/spaces/akshayyelne/discharge-intelligence-assistant


**📊 Future Enhancements**

  Potential improvements include:

  Automated discharge KPI scoring

  Predictive readmission risk models

  Visualization dashboards

  EMR system integration

  Real-time discharge monitoring

  Clinical cohort analysis

**📄 Documentation**

Detailed implementation documentation is available here:

docs/Patient_Discharge_Intelligence_Assistant_Project_Documentation.pdf

🎯 Use Case

This project demonstrates how hybrid LLM + RAG architectures can convert unstructured clinical documents into actionable insights that support:

  Hospital discharge optimisation

  Clinical decision support

  Risk identification

  Operational healthcare analytics

👨‍💻 Author

  AI Portfolio Project exploring:

  Healthcare AI

  Retrieval-Augmented Generation

  Clinical document intelligence

  LLM-powered healthcare analytics

⭐ If you find this project interesting, consider giving the repository a star.
