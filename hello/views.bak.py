from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

  
#===========================================================================
# This sample is Python code that summarizes a web page by GPT-4-32k. 
# The feature is designed to avoid SSL errors that occur when 
# accessing SSL websites from insideof Merck/MSD's intranet, 
# using BeutifulSoap, RecursiveCharacterTextSplitter, load_summarize_chain 
# and URLOpen library.
# This sample is written by Hiromu Okada, Learning and development,
# Oncology Product Sales Training, in Japan.
# library version: openai 0.28.1, langchain 0.0.305, beautifulsoup4 4.12.2,
#===========================================================================

# ==== Try https://(SSL)
target_url = 'https://en.wikipedia.org/wiki/Tsuru_no_Ongaeshi'
# ====
max_response_tokens = 1000
token_limit = 16000
# ====

import os, tiktoken
from bs4 import BeautifulSoup
from urllib.request import urlopen
from langchain.chat_models import AzureChatOpenAI
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
#from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
#from langchain.schema import AIMessage, HumanMessage, SystemMessage


def index(request):


  chat_GPT4_32k = AzureChatOpenAI(
      openai_api_type='azure',
      openai_api_base='https://iapi-test.merck.com/gpt/libsupport',
      openai_api_version='2023-05-15',
      #openai_api_key=gpteal_key,
      openai_api_key=os.getenv('XMerckAPIKey'),
      deployment_name='gpt-4-32k-0613',
      max_retries=1,
      max_tokens=max_response_tokens,
      #cache=True,
  )

  print('======')
  print('Please wait! Now loading URL as text')
  soup = BeautifulSoup( urlopen( target_url ).read().decode('utf-8', 'ignore'), 'html.parser' )
  long_text = soup.text
  while(long_text.find('\n\n')>=0):
    long_text = long_text.replace('\n\n','\n')
  enc = tiktoken.get_encoding('cl100k_base')
  num_of_tokens = len(enc.encode(long_text))
  cost = (num_of_tokens * 0.12) / 1000
  print('======')
  print('tokens={}, cost= ${} for loading web text '.format(num_of_tokens, cost))
  print('Please be cautious about over using of tokens!!')
  print('If you will waste token over limit of your authority, you will get message \"Requests exceeded token rate limit of your current OpenAI S0 pricing tier.\"')      
  print('======')
  #input('If you accept? push any key if you accept or CTRL+C to cancel')
  print('======')
  print('Please wait a moment!')
  text_splitter = RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=100) 
  texts = text_splitter.split_text(long_text)
  docs = [Document(page_content=t) for t in texts]

  template = '''Write a concise summary of the following:
  ```
  {text}
  ```
  CONCISE SUMMARY:'''

  prompt = PromptTemplate(template=template, input_variables=['text'])
  chain = load_summarize_chain( 
    llm= chat_GPT4_32k, 
    chain_type = 'map_reduce', 
    map_prompt = prompt, 
    combine_prompt = prompt, 
    #prompt = prompt,
    #verbose=True,
    verbose=False,
  )

#  short_text_obj = chain({'input_documents': docs},return_only_outputs=True)
#  print('======')
#  print('[summary]\n', short_text_obj)


  return HttpResponse("Hello Django!!")