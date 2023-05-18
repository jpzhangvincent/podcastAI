from langchain.prompts import PromptTemplate

map_prompt_template = """You are the expert podcast editor. Given the podcast transcript snippets, please write a concise summary 
with topic key words and separate by each paragraph if needed.

Content:
{text}

summary:"""
MAP_PROMPT = PromptTemplate(template=map_prompt_template, input_variables=["text"])

combine_prompt_template = """
Given the following list of texts, please extract the main topics(i.e less than 4 words) and paraphrase relevant texts belong to the same or semantically similar topics. 
Let's think step by step to iterate the list and make sure you include all the topics without duplicated topics.

Content:
{text}

Please output in the following json format without the `output_text` key and nested structure.
```
# output summary format:
{{
  "topic1": "paraphrased or summarized text",
  "topic2": "paraphrased or summarized text",
  ...
}}
```

summary:
"""
COMBINE_PROMPT = PromptTemplate(template=combine_prompt_template, input_variables=["text"])


paraphrase_template = """You are an expert youtube podcast editor. Please paraphrase and summarize the context
in a question format for semantic search later. You will only paraphrase and summarize when the speakers are talking 
about a specific subject. Return empty string if they are just chitchatting.

Context: {context}
Question: """
PARAPHRASE_PROMPT = PromptTemplate(input_variables=["context"], template=paraphrase_template)