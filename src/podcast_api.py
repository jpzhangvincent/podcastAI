import os
import pickle
import json 
from loguru import logger
from lcserve import serving
from langchain.chat_models import ChatOpenAI
from langchain_utils import get_summary, get_qa_with_sources
from data_utils import get_youtube_transcript, read_data_pickle
from typing import Dict

# def get_vectorstore(filename='../poc_app/combined_hf_faiss_vectorstore.pkl'):
#     print(f"Loading vectorstore from {filename}")
#     with open(filename, "rb") as f:
#         vectorstore = pickle.load(f)
#     return vectorstore

# vectorstore = get_vectorstore()

allin_youtube_episodes_df = read_data_pickle('../data/allin_youtube_episodes_df.pkl')
allin_faiss_index = read_data_pickle('../data/allin_faiss_index.pkl')


@serving
def get_summarized_topics(videoid:str, **kwargs) -> str:
    openai_api_key = os.environ['OPENAI_API_KEY']
    llm = ChatOpenAI(
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
    

@serving
def get_qa_search(querytext:str, **kwargs) -> Dict:
    openai_api_key = os.environ['OPENAI_API_KEY']
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model_name='gpt-3.5-turbo',
    )
    answer = get_qa_with_sources(querytext, llm, allin_faiss_index)
    return answer 