import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from prompts import MAP_PROMPT, COMBINE_PROMPT, PARAPHRASE_PROMPT
from langchain.chains import SequentialChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import LLMChain, LLMCheckerChain, RetrievalQAWithSourcesChain, LLMSummarizationCheckerChain

tokenizer = tiktoken.get_encoding('cl100k_base')
def tiktoken_len(text: str) -> int:
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def get_summary(transcript_txt):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,  # number of tokens overlap between chunks
            length_function=tiktoken_len,
            separators=['\n\n', '.\n', '\n', '.', '?', '!', ' ', '']
        )
    texts = text_splitter.split_text(transcript_txt)
    docs = [Document(page_content=t) for t in texts]

    openai_api_key = os.environ['OPENAI_API_KEY']
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model_name='gpt-3.5-turbo',
    )

    chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=False, 
                                map_prompt=MAP_PROMPT, combine_prompt=COMBINE_PROMPT)
    topic_summary = chain({"input_documents": docs}, return_only_outputs=True)
    topic_summary_txt = topic_summary['output_text'].strip()
    return topic_summary_txt

def get_qa_with_sources(question, faiss_index):
    openai_api_key = os.environ['OPENAI_API_KEY']
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model_name='gpt-3.5-turbo',
    )
    qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    qa = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain, 
                                 retriever=faiss_index.as_retriever(),
                                 return_source_documents = True)

    answer = qa({"question": question}, return_only_outputs=True)
    return answer


def get_in_context_search(timestamp, videoid, allin_youtube_episodes_df, faiss_index):
    in_context_text = allin_youtube_episodes_df[(allin_youtube_episodes_df.id.str.startswith(videoid)) &\
                  (timestamp > allin_youtube_episodes_df['start']) &\
                  (timestamp < allin_youtube_episodes_df['end_time'])].text.values[0]
    openai_api_key = os.environ['OPENAI_API_KEY']
    # This is an LLMChain to write a synopsis given a title of a play and the era it is set in.
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.2)
   
    context_paraphrase_chain = LLMChain(llm=llm, prompt=PARAPHRASE_PROMPT, output_key="question")

    # This is an LLMChain to do the semantic search
    qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    retrieval_qa_chain = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain, 
                                 retriever=faiss_index.as_retriever(),
                                 return_source_documents = True)

    # This is the overall chain where we run these two chains in sequence.
    in_context_search_chain = SequentialChain(
        chains=[context_paraphrase_chain, retrieval_qa_chain],
        input_variables=["context"],
        # Here we return multiple variables
        output_variables=["question", "answer"],
        verbose=True,
    )
    result = in_context_search_chain({"context": in_context_text})
    return result

def get_fact_check(query):
    openai_api_key = os.environ['OPENAI_API_KEY']
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
    checker_chain = LLMSummarizationCheckerChain.from_llm(llm, max_checks=2, verbose=True)
    print(checker_chain)
    check_output = checker_chain.run(query)
    return check_output
        