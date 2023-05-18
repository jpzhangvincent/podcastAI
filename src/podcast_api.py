import os
from loguru import logger
from lcserve import serving
from langchain_utils import get_summary, get_qa_with_sources, get_in_context_search
from data_utils import get_youtube_transcript, read_data_pickle
from typing import Dict

allin_youtube_episodes_df = read_data_pickle('../data/allin_youtube_episodes_df.pkl')
allin_faiss_index = read_data_pickle('../data/allin_faiss_index.pkl')


@serving
def get_summarized_topics(videoid:str, **kwargs) -> str:
    transcript = get_youtube_transcript(videoid)
    logger.info(f"Transcript: {transcript[:100]}...")
    if transcript:
        topic_summary = get_summary(transcript)
        return topic_summary
    else:
        return ''
    

@serving
def get_qa_search(querytext:str, **kwargs) -> Dict:
    answer = get_qa_with_sources(querytext, allin_faiss_index)
    return answer 


def get_context_search(timestamp: float, videoid:str,  **kwargs) -> Dict:
    answer = get_in_context_search(timestamp, videoid, allin_youtube_episodes_df, allin_faiss_index)
    return answer 