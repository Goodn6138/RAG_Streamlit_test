from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from moviepy.editor import VideoFileClip
from langchain.schema import Document
import speech_recognition as sr
import tempfile
import os

os.environ["COHERE_API_KEY"] = "API_KEY"

embd = CohereEmbeddings(user_agent = "langchain-agent")
urls = [
  #  "https://clarityconsultants.com/blog/what-about-the-10-80-10-principle-a-reflection-on-leaders-delegating#:~:text=Understanding%20Delegation&text=But%20the%20fact%20of%20the,10%20percent%20is%20the%20end.",
  #  "https://www.inc.com/justin-bariso/how-to-lead-how-to-scale-10-80-10-rule-emotional-intelligence.html",
#   "https://strategicdiscipline.positioningsystems.com/blog-0/build-an-elite-team-10-80-10-principle-above-the-line"
]

pdf = [
#   "/content/10X.pdf",
#  "/content/The Lean Startup - Erick Ries.pdf"
]

video_path = r"/content/How X-rays see through your skin - Ge Wang.mp4"

web_docs = [WebBaseLoader(url).load() for url in urls]
web_doc_list = [item for sublist in web_docs for item in sublist]


pdf_docs = [PyPDFLoader(doc).load() for doc in pdf]
pdf_doc_list = [item for sublist in pdf_docs for item in sublist]



# Extract audio from video and transcribe it
def transcribe_video(video_path, chunk_length=100):
    # Extract audio from video
    video = VideoFileClip(video_path)
    audio_path = r"extracted_audio.wav"
    video.audio.write_audiofile(audio_path)

    # Initialize recognizer
    recognizer = sr.Recognizer()
    timestamps_texts = []

    # Transcribe audio
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        duration = len(audio.frame_data) / audio.sample_rate / audio.sample_width
        for start_time in range(0, int(duration), chunk_length):
            start_frame = int(start_time * audio.sample_rate * audio.sample_width)
            end_frame = int((start_time + chunk_length) * audio.sample_rate * audio.sample_width)
            audio_chunk = sr.AudioData(audio.frame_data[start_frame:end_frame], audio.sample_rate, audio.sample_width)
            try:
                text = recognizer.recognize_google(audio_chunk)
                timestamps_texts.append((start_time, text))
            except sr.UnknownValueError:
                timestamps_texts.append((start_time, "[Unintelligible]"))
            except sr.RequestError as e:
                print(f"Error with the Speech Recognition service: {e}")
                break

    # Clean up audio file
    os.remove(audio_path)

    return timestamps_texts

transcriptions_with_timestamps = transcribe_video(video_path, chunk_length=100)

transcription_docs = [Document(page_content=text, metadata={'source': f'Video at {time} seconds'}) for time, text in transcriptions_with_timestamps]

all_docs_list = web_doc_list + pdf_doc_list #+ transcription_docs

#SPLIT
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)

all_splits = text_splitter.split_documents(all_docs_list)


#add to vector_store
vectorstore = Chroma.from_documents(
    documents = all_splits,
    embedding = embd,
    persist_directory = "./chroma_db"
)

retriver = vectorstore.as_retriever()


#the above code was vectorizing and initializing the retriever, the following code will be the finalisation 
