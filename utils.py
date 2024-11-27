from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from whisper.utils import get_writer




def text_split(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=20)
    context = "\n\n".join(str(p.page_content) for p in data)
    texts = text_splitter.split_text(context)
    return texts

def data_to_text(file_path):
    if file_path.split(".")[1] == "docx":
        loader = Docx2txtLoader(file_path)
        data = loader.load()
    elif file_path.split(".")[1] == "txt":
        loader = TextLoader(file_path)
        data = loader.load()
    elif file_path.split(".")[1] == "pdf":
        loader = PyPDFLoader(file_path)
        data = loader.load()
    elif file_path.split(".")[1] == "vtt":
        with open(file_path, 'r') as file: 
            contents = file.read() 
            data = [(Document(page_content=contents, metadata={"source":file_path}))]  
            print(data)  
    return data

def read_txt_file(filename): 
  with open(filename, 'r') as file: 
      contents = file.read() 
      doc = [(Document(page_content=contents, metadata={"source":filename}))]
      return doc
  
def audio_to_txt(file_path):
    import whisper
    from whisper.utils import get_writer


    model = whisper.load_model("base")
    audio = "uploads\\"+file_path.split("\\")[-1]
    print(audio)
    # D:\Poc open_AI\ai_sync\uploads\End To End LLM Project Using LLAMA 2- Open Source LLM Model From Meta.mp3
    
    result = model.transcribe(audio)
    output_directory = "./uploads/"


    # Save as a TXT file without any line breaks
    with open("transcription.txt", "w", encoding="utf-8") as txt:
        txt.write(result["text"])
    txt_writer = get_writer("vtt", output_directory)
    txt_writer(result, audio)


    # print(file_path)
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # model_id = "distil-whisper/distil-large-v2"
    # model = AutoModelForSpeechSeq2Seq.from_pretrained(
    #     model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    # )
    # model.to(device)
    # processor = AutoProcessor.from_pretrained(model_id)
    # whisper = pipeline(
    #     "automatic-speech-recognition",
    #     model=model,
    #     tokenizer=processor.tokenizer,
    #     feature_extractor=processor.feature_extractor,
    #     max_new_tokens=128,
    #     torch_dtype=torch_dtype,
    #     device=device,
    # )
    # audio = file_path
    # transcription = whisper(audio,
    #                     chunk_length_s=30,
    #                     stride_length_s=5,
    #                     batch_size=8)
    # print("audio process -- sucessfully")
    # result = transcription["text"]
    # print(result)
    # # output_directory = "./uploads"
    # txt_writer = get_writer("vtt", output_directory)
    # txt_writer(result, audio)
    return result
