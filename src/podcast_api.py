import os
import pickle
import json 
from loguru import logger
from lcserve import serving
from langchain.llms import OpenAI
from langchain_utils import get_summary
from data_utils import get_youtube_transcript

# def get_vectorstore(filename='../poc_app/combined_hf_faiss_vectorstore.pkl'):
#     print(f"Loading vectorstore from {filename}")
#     with open(filename, "rb") as f:
#         vectorstore = pickle.load(f)
#     return vectorstore

# vectorstore = get_vectorstore()

@serving
def get_summarized_topics(videoid:str, **kwargs) -> str:
    openai_api_key = os.environ['OPENAI_API_KEY']
    llm = OpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model_name='gpt-3.5-turbo',
    )
    transcript = get_youtube_transcript(videoid)
    logger.info(f"Transcript: {transcript[:100]}...")
    if transcript:
        topic_summary = get_summary(llm, transcript)
        return topic_summary
    else:
        return ''