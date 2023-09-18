import langchain
import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

api = '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'

model = HuggingFaceHub(huggingfacehub_api_token=api, repo_id = 'tiiuae/falcon-7b-instruct',
                       model_kwargs= {"temperature":0.6,"max_new_tokens":500})

template = """ You are an AI assistant, which is healpful and give polite answers.

{question}
"""

prompt=PromptTemplate(template=template, input_variables=['question'])
llm = LLMChain(prompt=prompt,llm=model)

@cl.langchain_factory(use_async=False)
def factory():
    prompt=PromptTemplate(template=template, input_variables=['question'])
    llm = LLMChain(prompt=prompt,llm=model)
    return llm

