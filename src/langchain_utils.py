import json
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import tiktoken
from langchain.docstore.document import Document
from prompts import MAP_PROMPT, COMBINE_PROMPT


tokenizer = tiktoken.get_encoding('cl100k_base')
def tiktoken_len(text: str) -> int:
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def get_summary(llm, transcript_txt):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,  # number of tokens overlap between chunks
            length_function=tiktoken_len,
            separators=['\n\n', '.\n', '\n', '.', '?', '!', ' ', '']
        )
    texts = text_splitter.split_text(transcript_txt)
    docs = [Document(page_content=t) for t in texts]

    chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=False, 
                                map_prompt=MAP_PROMPT, combine_prompt=COMBINE_PROMPT)
    topic_summary = chain({"input_documents": docs}, return_only_outputs=True)
    topic_summary_txt = topic_summary['output_text'].strip()
    return topic_summary_txt