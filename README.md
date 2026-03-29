# CodeMedix

AI-powered medical dictation and clinical report generation system built using Streamlit, Google Cloud Speech-to-Text, and LangChain.

## Overview

CodeMedix converts doctor audio dictations into structured medical reports using speech recognition and large language models.

The system performs:

* Medical speech-to-text transcription
* Audio preprocessing and chunking
* PDF report generation
* Retrieval-Augmented Generation (RAG)
* Clinical information extraction using Mistral-7B

This enables automated generation of structured clinical documentation from voice recordings.

---

## Features

* Secure user authentication system
* MP3 to FLAC conversion for accurate speech recognition
* Google Cloud Medical Speech-to-Text integration
* Automated transcript generation
* PDF clinical document generation
* FAISS vector database for document retrieval
* LangChain-based RAG pipeline
* HuggingFace Mistral-7B LLM integration
* Structured medical report extraction including:

  * Chief complaint
  * HPI
  * Medications
  * Allergies
  * Symptoms
  * Diagnosis
  * Follow-up plans
* Streamlit web interface

---

## Project Structure

```
medical-chatbot/

data/
    readme.txt
    test.pdf

vectorstore/

medical_dictation.py
medibot.py
create_memory_for_llm.py
connect_memory_with_llm.py
hash_pass.py

config.yaml
counter.txt
Pipfile
Pipfile.lock
README.md
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/codemedix.git
cd codemedix
```

---

### 2. Install Dependencies

Using pipenv:

```bash
pipenv install
pipenv shell
```

Or using pip:

```bash
pip install streamlit streamlit-authenticator google-cloud-speech pydub fpdf langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers python-dotenv
```

---

## Environment Variables

Create a `.env` file in the root directory:

```
HF_TOKEN=your_huggingface_token
GOOGLE_APPLICATION_CREDENTIALS=codemedix-credentials.json
```

---

## Google Cloud Setup

1. Go to Google Cloud Console
2. Enable Speech-to-Text API
3. Create a Service Account
4. Download JSON credentials
5. Place the JSON file in the project folder
6. Set environment variable

### Mac/Linux

```bash
export GOOGLE_APPLICATION_CREDENTIALS="codemedix-credentials.json"
```

### Windows

```bash
set GOOGLE_APPLICATION_CREDENTIALS=codemedix-credentials.json
```

---

## HuggingFace Setup

1. Go to https://huggingface.co/settings/tokens
2. Create a new token
3. Add it to `.env`

```
HF_TOKEN=your_token_here
```

---

## Authentication Setup

Create `config.yaml`

```yaml
credentials:
  usernames:
    doctor:
      email: doctor@gmail.com
      name: Doctor
      password: hashed_password

cookie:
  name: codemedix_cookie
  key: some_secret_key
  expiry_days: 1
```

Generate hashed password:

```bash
python hash_pass.py
```

---

## Running the Application

```bash
streamlit run medical_dictation.py
```

---

## Usage

1. Login using credentials
2. Upload MP3 medical dictation
3. System converts audio to transcript
4. PDF is generated
5. LLM processes medical content
6. Structured clinical report is displayed

---

## Tech Stack

* Python
* Streamlit
* Google Cloud Speech-to-Text
* LangChain
* FAISS
* HuggingFace Mistral-7B
* PyDub
* FPDF
* YAML Authentication
* Sentence Transformers

---

## How It Works

```
MP3 Audio
   ↓
MP3 to FLAC Conversion
   ↓
Audio Chunking
   ↓
Google Speech-to-Text
   ↓
Transcript Generation
   ↓
PDF Creation
   ↓
Text Chunking
   ↓
Embeddings (MiniLM)
   ↓
FAISS Vector Store
   ↓
Mistral-7B (LangChain RAG)
   ↓
Structured Medical Report
```

