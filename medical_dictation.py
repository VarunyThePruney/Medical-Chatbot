import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
from google.cloud import speech
from pydub import AudioSegment
from fpdf import FPDF
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA

# Load YAML config for auth
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )

st.markdown(
    """
    <style>
    .title {
        text-align: center;
    }
    </style>
    <h1 class="title">CodeMedix </h1>
    """,
    unsafe_allow_html=True
)

name, auth_status, username = authenticator.login("Login", "main")


client = speech.SpeechClient()

client = speech.SpeechClient.from_service_account_file(credentials_path)

# Function to convert mp3 to flac (since Google Speech API works better with FLAC/WAV)
def convert_mp3_to_flac(mp3_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    flac_file_path = mp3_file_path.replace(".mp3", ".flac") #format for transcribing mp3 audio to text
    audio.export(flac_file_path, format="flac")
    return flac_file_path

# Function to split the audio file into chunks
def split_audio(input_file, chunk_duration_ms=60000):
    """Split the input audio file into smaller chunks."""
    audio = AudioSegment.from_file(input_file)
    chunks = []
    start_time = 0
    while start_time < len(audio):
        end_time = min(start_time + chunk_duration_ms, len(audio))
        chunk = audio[start_time:end_time]
        chunk_file = f"chunk_{start_time}.flac"
        chunk.export(chunk_file, format="flac")
        chunks.append(chunk_file)
        start_time = end_time
    return chunks

# Function to transcribe audio chunk directly to Gcp Speech-to-Text
def transcribe_audio_chunk(chunk_file):
    """Transcribe audio chunk directly to Google Cloud Speech-to-Text."""
    with open(chunk_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    medical_terms = [
        "hypertension", "diabetes", "stroke", "glucose", "insulin",
        "cardiovascular", "neurology", "epilepsy", "seizure", "radiology",
        "medication", "prescription", "surgical", "oncology", "chemotherapy"
    ]
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=48000,
        language_code="en-US",
        model="medical",  # Use the medical model
        speech_contexts=[speech.SpeechContext(
                phrases=medical_terms)
        ]
    )

    # Send the audio data to Google Cloud Speech API for recognition
    response = client.recognize(config=config, audio=audio)

    # Extract and return the transcript
    transcripts = []
    for result in response.results:
        transcripts.append(result.alternatives[0].transcript)

    return "\n".join(transcripts)

def get_file_number():
    counter_file = "counter.txt"

    # Ensure counter file exists, or create it with a starting value of 1
    if not os.path.exists(counter_file):
        with open(counter_file, "w") as f:
            f.write("1")
        return 1

    try:
        with open(counter_file, "r") as f:
            content = f.read().strip()

        last_number = int(content) if content.isdigit() else 0
    except Exception as e:
        print(f"Error reading counter file: {e}")
        last_number = 0

    new_number = last_number + 1

    with open(counter_file, "w") as f:
        f.write(str(new_number))

    return new_number


def generate_pdf(transcript):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, transcript)

    file_number = get_file_number()

    pdf_file = f"{file_number}_claim.pdf"
    pdf.output(pdf_file)

    return pdf_file


def save_pdf(content, file_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.multi_cell(190, 10, txt=content, align='L')

    # Save the PDF to the specified file path
    pdf.output(file_path)

DB_FAISS_PATH= "vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm

# Streamlit interface
def main():
    if auth_status is False:
        st.error("Authentication Failed")

    if auth_status:
        st.sidebar.success(f"Welcome, {name}!")
        authenticator.logout("Logout", "sidebar")




        st.title("Speech-to-Text Medical Dictation")

        # File uploader widget
        audio_file = st.file_uploader("Upload your MP3 file", type=["mp3"])

        if audio_file is not None:
            # Save the uploaded file to disk
            with open("temp_audio.mp3", "wb") as f:
                f.write(audio_file.read())

            st.success("File uploaded successfully!")

            # Convert MP3 to FLAC
            st.write("Converting MP3 to FLAC...")
            flac_file = convert_mp3_to_flac("temp_audio.mp3")

            # Split the audio file into chunks
            st.write("Splitting the audio file into chunks...")
            chunks = split_audio(flac_file)

            st.write(f"Audio file split into {len(chunks)} chunks.")

            full_transcript = ""

            # Process each chunk and transcribe
            for chunk_file in chunks:
                st.write(f"Processing chunk: {chunk_file}...")
                transcript = transcribe_audio_chunk(chunk_file)
                full_transcript += transcript + "\n\n"

                os.remove(chunk_file)

            # Show the full transcription
            st.subheader("Full Transcript:")
            st.text_area("Transcription", full_transcript, height=600)

            pdf_file = generate_pdf(full_transcript)

            os.makedirs("data", exist_ok=True)
            save_pdf(full_transcript, "data/test.pdf")
            st.subheader("Processing through LLM:")

            from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS

            from dotenv import load_dotenv, find_dotenv

            load_dotenv(find_dotenv())

            # Step 1: Load raw PDF(s)
            DATA_PATH = "data/"

            def load_pdf_files(data):
                loader = DirectoryLoader(data,
                                         glob='*.pdf',
                                         loader_cls=PyPDFLoader)

                documents = loader.load()
                return documents

            documents = load_pdf_files(data=DATA_PATH)
            print(documents)


            # Step 2: Create Chunks
            def create_chunks(extracted_data):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                               chunk_overlap=50)
                text_chunks = text_splitter.split_documents(extracted_data)
                return text_chunks

            text_chunks = create_chunks(extracted_data=documents)


            def get_embedding_model():
                embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                return embedding_model

            embedding_model = get_embedding_model()

            # Step 4: Store embeddings in FAISS
            DB_FAISS_PATH = "vectorstore/db_faiss"
            db = FAISS.from_documents(text_chunks, embedding_model)
            os.makedirs("vectorstore", exist_ok=True)
            db.save_local(DB_FAISS_PATH)

            # Clean up
            os.remove("temp_audio.mp3")
            os.remove(flac_file)
            os.remove(pdf_file)


            if 'messages' not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                st.chat_message(message['role']).markdown(message['content'])

            #16 prompts for the LLM
            prompts = ["""Generate a concise chief complaint for a medical transcription report based on a patient's primary concern during their visit. If none exist, just return "None are present." Do not make it past 1 sentence and keep it short and concise, also do not include the patient’s details or name in the response.""",

    """Write a detailed history of present illness (HPI) for a specific condition of the patient, including onset, duration, associated symptoms, relieving and aggravating factors, and any treatment attempted. If none exist, just return "None are present." Do not make it past 1 sentence and keep it short and concise, also do not include the patient’s details or name in the response.""",

    """List the past medical history of a patient, including chronic conditions, previous surgeries, and any significant hospitalizations. If none exist, just return "None are present." Do not make it past 1 sentence and keep it short and concise, also do not include the patient’s details or name in the response.""",

    """Generate a list of medications for a patient that treats a specific condition, including dosage and frequency, while ensuring accuracy in medical terminology. Only list medication, not the purpose of the medication. Also, do not include the patient’s details or name in the response.""",

    """Provide a list of known allergies for a patient, including drug, food, and environmental allergens, along with any documented reactions. If none exist, just return "None are present." Do not make it past 1 sentence and keep it short and concise, also do not include the patient’s details or name in the response.""",

    """Summarize a patient's family medical history, including hereditary conditions such as heart disease, diabetes, or cancer, mentioning and emphasizing immediate family members. If none exist, just return "None are present." Do not make it past 1 sentence and keep it short and concise, also do not include the patient’s details or name in the response.""",

    """List general symptoms reported by the patient. This may include sickness like fever or common cold, or physical symptoms like fatigue or weight loss. If none exist, just return "None are present." Do not make it past 1 sentence and keep it short and concise, also do not include the patient’s details or name in the response.""",

    """Describe in a concise way any respiratory symptoms related to the patient such as cough, shortness of breath, wheezing, or crackling. If none exist, just return "None are present." Do not make it past 1 sentence and keep it short and concise, also do not include the patient’s details or name in the response.""",

    """Describe in a concise way any cardiovascular-related symptoms related to the patient, including chest pain, palpitations, or leg swelling. If none exist, just return "None are present." Do not make it past 1 sentence and keep it short and concise, also do not include the patient’s details or name in the response.""",

    """Describe in a concise way any gastrointestinal complaints such as nausea, vomiting, abdominal pain, or bowel habit changes. If none exist, just return "None are present." Do not make it past 1 sentence and keep it short and concise, also do not include the patient’s details or name in the response.""",

    """List in a concise way any neurological symptoms like headaches, dizziness, numbness, or weakness. If none exist, just return "None are present." Do not make it past 1 sentence and keep it short and concise, also do not include the patient’s name or details in the response.""",

    """Generate a summary of the patient's vital signs, including temperature, blood pressure, heart rate, respiratory rate, and oxygen saturation. If none exist, just return "None are present." Do not make it past 1 sentence and keep it short and concise, also do not include the patient’s name or details in the response.""",

    """Generate a summary of abdominal findings, including tenderness, masses, or organ enlargement. If none exist, just return "None are present." Do not make it past 1 sentence and keep it short and concise, also do not include the patient’s name or details in the response.""",

    """Provide the primary diagnosis and differential diagnoses based on the patient’s symptoms and examination. If none exist, just return "None are present." Do not make it past 1 sentence and keep it short and concise, also do not include the patient’s name or details in the response.""",

    """List prescribed medications for the diagnosis with dosage, frequency, and duration. Only include medication that was prescribed this visit, not any that was prescribed previously. If none exist, just return "None are present." """,

    """Suggest follow-up plans, including next visit schedule, referrals, or additional testing. If none exist, just return "None are present." Do not make it past 1 sentence and keep it short and concise, also do not include the patient’s name or details in the response."""
]
            responses = []
            for prompt in prompts:
                if prompt:
                    CUSTOM_PROMPT_TEMPLATE = """
                            Use the pieces of information provided in the context to answer user's question.
                            If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                            Dont provide anything out of the given context

                            Context: {context}
                            Question: {question}

                            Start the answer directly. No small talk please.only give answer
                            """

                    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
                    HF_TOKEN = os.environ.get("HF_TOKEN")
                    if HF_TOKEN is None:
                        st.error("HF_TOKEN not found in environment variables")
                        return

                    try:
                        vectorstore = db
                        if vectorstore is None:
                            st.error("Failed to load the vector store")

                        qa_chain = RetrievalQA.from_chain_type(
                            llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                            chain_type="stuff",
                            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                            return_source_documents=True,
                            chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                        )

                        response = qa_chain.invoke({'query': prompt})

                        result = response["result"]
                        responses.append(result)


                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            formatted_responses = "\n\n".join([f"{i + 1}. {response}" for i, response in enumerate(responses)])

            st.chat_message('user').markdown(formatted_responses)
            st.session_state.messages.append({'role': 'user', 'content': formatted_responses})


if __name__ == "__main__":
    main()
