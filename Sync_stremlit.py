# import streamlit as st
# import os
# import urllib
# import warnings
# from pathlib import Path as p
# from pprint import pprint
# import pandas as pd

# from langchain.chains.question_answering import load_qa_chain
# from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader
# # from langchain.vectorstores import Chroma
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.summarize import load_summarize_chain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain import PromptTemplate
# from utils import data_to_text
# from utils import audio_to_txt
# from prompt import (map_prompt_template,combine_prompt_template,map_prompt_template2,combine_prompt_template2,
# map_prompt_template3,combine_prompt_template3,map_prompt_template4,combine_prompt_template4,
# map_prompt_template5,combine_prompt_template5,map_prompt_template6,combine_prompt_template6,
# map_prompt_template7,combine_prompt_template7,map_prompt_template8,combine_prompt_template8)
# import warnings
# warnings.filterwarnings("ignore")
# import google.generativeai as genai
# import torch


# GOOGLE_API_KEY="AIzaSyCruOjusCfgSYI7CMsr_7u_uFq8JMR9RtQ"

# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",google_api_key=GOOGLE_API_KEY,
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )

import streamlit as st
import os
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint
import pandas as pd
import asyncio

from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from utils import data_to_text
from utils import audio_to_txt
from prompt import (
    map_prompt_template, combine_prompt_template, map_prompt_template2, combine_prompt_template2,
    map_prompt_template3, combine_prompt_template3, map_prompt_template4, combine_prompt_template4,
    map_prompt_template5, combine_prompt_template5, map_prompt_template6, combine_prompt_template6,
    map_prompt_template7, combine_prompt_template7, map_prompt_template8, combine_prompt_template8
)
import warnings
warnings.filterwarnings("ignore")
import google.generativeai as genai
import torch

# Ensure there is an event loop running
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

GOOGLE_API_KEY = "AIzaSyCruOjusCfgSYI7CMsr_7u_uFq8JMR9RtQ"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


# genai.configure(api_key=GOOGLE_API_KEY)

# for m in genai.list_models():
#   if 'generateContent' in m.supported_generation_methods:
#     print(m.name)

# Create a folder if it doesn't exist
# GOOGLE_API_KEY="AIzaSyCruOjusCfgSYI7CMsr_7u_uFq8JMR9RtQ"


# model = ChatGoogleGenerativeAI(model="gemini-1.5-pro",google_api_key=GOOGLE_API_KEY,
#                              temperature=0.5,convert_system_message_to_human=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",google_api_key=GOOGLE_API_KEY,
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


upload_folder = "uploads"
os.makedirs(upload_folder, exist_ok=True)

# Step 1: File upload
uploaded_file = st.file_uploader("Upload a file", type=["vtt", "txt", "docx","pdf","mp3"])
print("sssssssssssssssssss",uploaded_file)

# Save the uploaded file to the upload folder
if uploaded_file is not None:
    file_path = os.path.join(upload_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write(f"File saved to: {file_path}")
    if uploaded_file.name.split(".")[1]=="mp3":
        print(f"File saved to: {file_path}")
        filename = "uploads\\"+file_path.split("\\")[1]
        print(filename)
        data = audio_to_txt(filename)
    else:
        data = data_to_text(file_path)
    # print(data)
# Step 2: Select feature
    if uploaded_file.name.split(".")[1]=="mp3":
        file_path = os.path.join(upload_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write(f"File saved to: {file_path}")
        data = audio_to_txt(file_path)
    


selected_feature = st.selectbox(
    "Select a feature",
    ("Summary", "KDD", "User story","Sentiment analysis","emotion analysis","commitment analysis","wsr","raid log")
)

# Proceed with selected feature
if st.button("Proceed"):
    if selected_feature == "Summary":
        summary_chain = load_summarize_chain(llm=model,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template,
                                    )
        output = summary_chain.run(data)
        st.write(output)
        # output_file_path = os.path.join(upload_folder,"output.txt")
        # with open(output_file_path, "w") as f:
        #     f.write(output)

    elif selected_feature == "KDD":
        summary_chain = load_summarize_chain(llm=model,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template2,
                                     combine_prompt=combine_prompt_template2,
                                    )
        output = summary_chain.run(data)
        st.write(output)
    elif selected_feature == "User story":
        summary_chain = load_summarize_chain(llm=model,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template3,
                                     combine_prompt=combine_prompt_template3,
                                    )
        output = summary_chain.run(data)
        st.write(output)
    elif selected_feature == "Sentiment analysis":
        summary_chain = load_summarize_chain(llm=model,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template4,
                                     combine_prompt=combine_prompt_template4,
                                    )
        output = summary_chain.run(data)
        st.write(output)

    elif selected_feature == "emotion analysis":
        summary_chain = load_summarize_chain(llm=model,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template5,
                                     combine_prompt=combine_prompt_template5,
                                    )
        output = summary_chain.run(data)
        st.write(output)

    elif selected_feature == "commitment analysis":
        summary_chain = load_summarize_chain(llm=model,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template6,
                                     combine_prompt=combine_prompt_template6,
                                    )
        output = summary_chain.run(data)
        st.write(output)
    

    elif selected_feature == "wsr":
        summary_chain = load_summarize_chain(llm=model,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template7,
                                     combine_prompt=combine_prompt_template7,
                                    )
        output = summary_chain.run(data)
        st.write(output)

    elif selected_feature == "raid log":
        summary_chain = load_summarize_chain(llm=model,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template8,
                                     combine_prompt=combine_prompt_template8,
                                    )
        output = summary_chain.run(data)
        st.write(output)
    
        # print("print")
        


question = st.text_input("Ask your question 👇")
# Display the user input
if st.button("Doc chat"):
    if question:
        GOOGLE_API_KEY="AIzaSyCruOjusCfgSYI7CMsr_7u_uFq8JMR9RtQ"

        model = ChatGoogleGenerativeAI(model="gemini-1.0-pro",google_api_key=GOOGLE_API_KEY,
                                    temperature=0.3,convert_system_message_to_human=True)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
        data = data_to_text(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        content = "\n\n".join(str(page.page_content) for page in data)
        texts = text_splitter.split_text(content)
        print(len(texts))

        # texts = text_split(data)  # Assuming text_split is used to split the question into sentences or tokens
        # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vector_index = FAISS.from_documents(data, embeddings).as_retriever()
        docs = vector_index.get_relevant_documents(question)
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        # model = ChatGoogleGenerativeAI(model="gemini-pro",
        #                      temperature=0.3)

        prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        
        response = chain(
            {"input_documents":docs, "question": question}
            , return_only_outputs=True)
        # qa_chain = RetrievalQA.from_chain_type(
        #     model,
        #     retriever=vector_index,
        #     return_source_documents=True
        # )
        

        # result = qa_chain({"query": question})
        st.write("Chat Bot response:")
        st.write(response["output_text"])



    # question = st.text_area("Ask your question 👇")
    # if question:
    #     data = data_to_text(file_path)
    #     texts = text_split(data)  # Assuming text_split is used to split the question into sentences or tokens
    #     vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})
    #     qa_chain = RetrievalQA.from_chain_type(
    #         model,
    #         retriever=vector_index,
    #         return_source_documents=True
    #     )
    #     result = qa_chain({"query": question})

    #     st.write("You entered:", question)
    #     st.write("Chat Bot response:")
    #     st.write(result["result"])



# Chat bot
# elif st.button("Chat Bot"):
#     print("ksdjskjsjkd")
#     question = st.text_input("Ask your question")
#     st.write("You entered: ", question)
#     print(question)
#     if question:
#         data = data_to_text(file_path)
#         # print(data)
#         print("jskdjksdjskdjsdjskjdksjdjsdksjdskjdksdjskdj")
#         texts = text_split(data)
#         vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})
#         qa_chain = RetrievalQA.from_chain_type(
#         model,
#         retriever=vector_index,
#         return_source_documents=True
#         )
#         result = qa_chain({"query": question})

#         st.write(result["result"])
        # Your chat bot logic here, e.g., using a pre-trained model
