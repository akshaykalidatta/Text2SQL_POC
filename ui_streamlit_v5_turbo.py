from langchain.document_loaders import CSVLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings, VertexAIEmbeddings
from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel
from langchain.chat_models import ChatOpenAI
from langchain.llms import VertexAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain import hub
import time
import pickle
import os
import vertexai
from vertexai.language_models import CodeGenerationModel
from vertexai.preview.language_models import TextGenerationModel
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import openai
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatVertexAI
import argparse
from langchain.embeddings import VertexAIEmbeddings
from langchain.document_loaders import JSONLoader
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from google.cloud import bigquery
from typing import List
from tqdm import tqdm
import logging
import json
import os 
import sys
from openai import OpenAI
import plotly.express as px
import streamlit as st
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.prompts.chat import AIMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from vertexai.preview.generative_models import GenerativeModel
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

# -------------------------------------------------------------------------------

os.environ["OPENAI_API_KEY"] = ""
openai_api_key = os.environ.get('OPENAI_API_KEY')

#-------------------------------------------------------------------------------


messages = []
vertex_template = "You are a Bigquery expert capable of writing complex SQL query in Bigquery. Use only  bigquery  native functions and construct."
system_message_prompt = SystemMessagePromptTemplate.from_template(vertex_template)
messages.append(system_message_prompt)
human_template = """Given the following inputs:
----
USER_QUERY:{query}
---
TABLES : {matched_tables}
---
MATCHED_SCHEMA: {matched_schema}
------
IMPORTANT: Use ONLY the column names (column_name) mentioned in MATCHED_SCHEMA with the table_name. DO NOT USE any other column names outside of this. 
IMPORTANT: Associate column_name mentioned in MATCHED_SCHEMA only to the table_name specified under MATCHED_SCHEMA
IMPORTANT: avoid using alias column in where clause . 
IMPORTANT: use safe divide , safe cast where ever possible
"""
human_message = HumanMessagePromptTemplate.from_template(human_template)
messages.append(human_message)
vertex_prompt = ChatPromptTemplate.from_messages(messages)

#--------------------------------------
gemini_prompt="""
"You are a Bigquery expert capable of writing complex SQL query in Bigquery. Use only  bigquery  native functions and constructs.
--- given the table details with their description and columns used for joins . which are '|' seperated .
``
TABLES : {matched_tables}
``
---
given the column details, description and usage which are '|' seperated as following . 
```
MATCHED_SCHEMA: {matched_schema}


```
Write a GoogleSQL dialect SQL using the above column details and tables provided in details which achieves the following.
```
{content}
```
IMPORTANT: Use ONLY the column names (column_name) mentioned in MATCHED_SCHEMA. DO NOT USE any other column names outside of this. 
IMPORTANT: Associate column_name mentioned in MATCHED_SCHEMA only to the table_name specified under MATCHED_SCHEMA.
Note: Use SAFE_DIVIDE, SAFE_CAST wherever possible. Instead of using strftime(), use BigQuery native functions like EXTRACT(YEAR FROM date_column) and EXTRACT(MONTH FROM date_column) for date handling. Avoid using alias names in GROUP BY and ORDER BY sections.
Return only the SQL query generated as output. Don't return SQL if the context is not transaction or customer related.

"""

# -------------------------------------------------------------------------------

class MyVertexAIEmbeddings(VertexAIEmbeddings, Embeddings):
    model_name = 'textembedding-gecko'
    max_batch_size = 5

# -------------------------------------------------------------------------------

    def embed_segments(self, segments: List) -> List:
        embeddings = []
        for i in tqdm(range(0, len(segments), self.max_batch_size)):
            batch = segments[i: i+self.max_batch_size]
            embeddings.extend(self.client.get_embeddings(batch))
        return [embedding.values for embedding in embeddings]


# -------------------------------------------------------------------------------

    def embed_query(self, query: str) -> List:
        embeddings = self.client.get_embeddings([query])
        return embeddings[0].values

embedding = MyVertexAIEmbeddings()

# -------------------------------------------------------------------------------    

def llama_init():
    n_gpu_layers = 1  # Metal set to 1 is enough.
    n_batch = 1024  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Make sure the model path is correct for your system!
    llm_llama = LlamaCpp(
        model_path="llama.cpp/models/7B/ggml-model-f16.bin",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=4096,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=True,
    )
    return llm_llama

# -------------------------------------------------------------------------------    

def describe_bot(prompt,llm_type):
    if llm_type == "palm2":
        vertexai.init(project="dmgcp-del-136", location="us-central1")
        parameters = {
            "max_output_tokens": 1024,
            "temperature": 0,
            "top_p": 0.8,
            "top_k": 40
        }
        model = TextGenerationModel.from_pretrained("text-bison@001")
        response = model.predict(
            f"""{prompt}""",
            **parameters
        )
        return response.text
        pass
    
    elif llm_type == 'openai':
        client = OpenAI()
        OpenAI.api_key = os.getenv('OPENAI_API_KEY')
        completion = client.completions.create(
          model="gpt-3.5-turbo-instruct",
          prompt= """{}""".format(prompt),
          max_tokens=3000
        )
        return completion.choices[0].text
        pass
    
    elif llm_type == 'gemini-pro':
        MODEL_NAME = 'gemini-pro'
        model = GenerativeModel(MODEL_NAME)
        responses = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 800,
                "top_p": 1.0,
                "top_k": 40,
            },
            stream=False
        )
        response = '\n'.join(responses.text.strip().replace('```sql',' ').replace('```',' ').split('\n'))
        return response
        pass
    elif llm_type == 'llama':
        llm_llama =llama_init()
        # print('reached desc')
        answer = llm_llama(prompt)
        # print('reached desc ans')
        # print(answer)
        return answer
        pass

# -------------------------------------------------------------------------------    

def predict(content: str,schema: str,tables:str,llm):
    
    verbose: bool = False
    TEMPLATE = f'''
    --- given the table details with description and columns used for joins . which are '|' seperated .
    ``
    TABLES : {tables}
    ``
    ---
    given the column details which are '|' seperated as following . 
    ```
    MATCHED_SCHEMA: {schema}


    ```
    Write a GoogleSQL dialect SQL using the above column details and tables provided in details which achieves the following.
    ```
    { content }
    ```
    IMPORTANT: Use ONLY the column names (column_name) mentioned in MATCHED_SCHEMA. DO NOT USE any other column names outside of this. 
    IMPORTANT: `` Associate column_name mentioned in MATCHED_SCHEMA only to the table_name specified under MATCHED_SCHEMA ``.
    Note : use safe divide , safe cast where ever possible. Avoid using alias names in groupby and orderby sections.
    '''

    client = OpenAI()
    
    response = client.chat.completions.create(
        model = "gpt-4-0125-preview",
        messages = [{"role": "user", "content": f"{TEMPLATE}"}]
    )
    
    return response.choices[0].message.content.strip()
    pass

# -------------------------------------------------------------------------------    

def return_tables_matched(query):
    
    documents = JSONLoader(file_path='tables.jsonl', jq_schema='.', text_content=False, json_lines=True).load()
    db = FAISS.from_documents(documents=documents, embedding=embedding)
    retriever = db.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'lambda_mult': 1})
    matched_documents = retriever.get_relevant_documents(query=query)
    matched_tables = []
    for document in matched_documents:
        page_content = document.page_content
        page_content = json.loads(page_content)
        table_name = page_content['table_name']
        desc=page_content['description']
        matched_tables.append(f'{table_name}|{desc}')
    matched_tables = '\n'.join(matched_tables)    
    return matched_tables
    pass

# -------------------------------------------------------------------------------    

def return_matched_columns(query):
    
    documents = JSONLoader(file_path='columns_new.jsonl', jq_schema='.', text_content=False, json_lines=True).load()
    db = FAISS.from_documents(documents=documents, embedding=embedding)
    search_kwargs = {
        'k': 20
    }
    retriever = db.as_retriever(search_type='similarity', search_kwargs=search_kwargs)
    matched_columns = retriever.get_relevant_documents(query=query)
    
    return matched_columns
    pass

# -------------------------------------------------------------------------------    

def return_columns_cleaned(query):
    
    matched_columns = return_matched_columns(query)
    
    matched_columns_filtered = []
    # LangChain filters does not support multiple values at the moment
    for i, column in enumerate(matched_columns):
        page_content = json.loads(column.page_content)
        matched_columns_filtered.append(page_content)

    matched_columns_cleaned = []
    for doc in matched_columns_filtered:
        table_name = doc['table_name']
        column_name = doc['column_name']
        data_type = doc['data_type']
        col_desc = doc['description']
        matched_columns_cleaned.append(f'table_name={table_name}|column_name={column_name}|data_type={data_type}|description={col_desc}') 
    matched_columns_cleaned = '\n'.join(matched_columns_cleaned)
    return matched_columns_cleaned
    pass

# -------------------------------------------------------------------------------    

def run_on_bq(sql):
    print(sql)
    job_config = bigquery.QueryJobConfig(default_dataset="dmgcp-del-136.demo_data_20231221")
    bq = bigquery.Client()
    df = bq.query(sql,job_config).to_dataframe()
    print('type',type(df))
    return df
    pass

#-------------------------------------------------------------------------------    
def get_sql_query_openai(query,llm):
    columns_schema = return_columns_cleaned(query)
    tables= return_tables_matched(query)
    result=predict(query,columns_schema,tables,llm)
    
    output_lst = result.split('```')
    if len(output_lst)> 1:
        sql =  output_lst[1]
    else:
        sql = output_lst[0]
        
    cleaned_sql=sql.replace('sql','')
    return cleaned_sql
    pass
#------------------------------------------------------------------------------- 
def get_sql_query_vertexai(query,llm):
    columns_schema = return_columns_cleaned(query)
    tables= return_tables_matched(query)
    request = vertex_prompt.format_prompt(query=query,matched_schema=columns_schema,matched_tables=tables)
    response = llm(request.to_messages())
    sql = '\n'.join(response.content.strip().split('\n')[1:-1])
    return sql

#------------------------------------------------------------------------------- 
def get_sql_query_geminiai(query,MODEL_NAME):
    columns_schema = return_columns_cleaned(query)
    tables = return_tables_matched(query)
    prompt = gemini_prompt.format(content=query,matched_tables=tables,matched_schema=columns_schema)
    model = GenerativeModel(MODEL_NAME)
    responses = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.0,
            "max_output_tokens":2048,
        },
        stream=False
    )
    sql = '\n'.join(responses.text.strip().replace('```sql',' ').replace('```',' ').split('\n'))
    return sql

#-------------------------------------------------------------------------------#

def generate_visualization(df, color_scheme='Plasma'):
    cols = df.columns 
    col_x = cols[-2]
    col_y = cols[-1]
    
    print("X Column data type:", df[col_x].dtype)
    print("Y Column data type:", df[col_y].dtype)
    
    # Determine the type of plot based on the data type of the columns
    if df[col_x].dtype == 'object' or df[col_y].dtype == 'object':
        # Bar plot for categorical data
        fig = px.bar(df, x=col_x, y=col_y, title='Insight Visualization', color=col_y, color_continuous_scale=color_scheme)
    elif df[col_x].dtype == 'datetime':
        # Line plot for time series data
        fig = px.line(df, x=col_x, y=col_y, title='Insight Visualization', color=col_y, color_continuous_scale=color_scheme)
    else:
        # Scatter plot for numerical data
        fig = px.scatter(df, x=col_x, y=col_y, title='Insight Visualization', color=col_y, color_continuous_scale=color_scheme)
    
    fig.show()
    fig.write_image("image_desc.png")
#-------------------------------------------------------------------------------
def sample_gen(df,llm_type):
    if llm_type == 'llama':
        sample = describe_bot(f"""
                    <s>[INST] <<SYS>>
                    You are a helpful assistant and based on the context answer users question.
                    context :  values:{df}, metadeta:{md_map}
                    <</SYS>>
        
                     and give me a top 5 insights of this data in bullet points on seperate lines.
                    [/INST]
                    """,llm_type)
    else:
        sample = describe_bot(f'''give me top 3 sample questions looking at this data: {df} and dont highlight the words or give any headings. just 3 questions in seperate lines without numbering and nothing else''',llm_type)
    return sample
    pass
#-------------------------------------------------------------------------------
def main():
    
    st.title('Automated BI: Text2Insights 📈')
    main_placeholder = st.empty()
    question = st.text_input('Enter your question:')
    llm_type = st.selectbox('Select LLM Type:', ['openai', 'palm2','gemini-pro'])
    sql_query=''
    if st.button('Generate'):
        try:
            if llm_type =='openai':
                main_placeholder.text(f"!!  Running on OpenAI!!")
                llm=ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
                sql_query = get_sql_query_openai(question,llm)
                
            elif llm_type =='palm2':
                main_placeholder.text(f"!!  Running on Palm2 !!")
                PROJECT = 'dmgcp-del-136'
                LOCATION = 'us-central1'
                MODEL_NAME = 'codechat-bison@002'
                llm = ChatVertexAI(project=PROJECT,location=LOCATION,model_name=MODEL_NAME,
                                   temperature=0.0,max_output_tokens=2048)
                sql_query = get_sql_query_vertexai(question,llm)
                
            elif llm_type =='gemini-pro':
                main_placeholder.text(f"!!  Running on Gemini-Pro LLM !!")
                PROJECT = 'dmgcp-del-136'
                LOCATION = 'us-central1'
                MODEL_NAME = "gemini-pro"
                sql_query = get_sql_query_geminiai(question,MODEL_NAME)
                
            elif llm_type =='llama':
                main_placeholder.text(f"!!  Running on Llama 2 LLM !!")
                PROJECT = 'dmgcp-del-136'
                LOCATION = 'us-central1'
                MODEL_NAME = 'codechat-bison@002'
                llm = ChatVertexAI(project=PROJECT,location=LOCATION,model_name=MODEL_NAME,
                                   temperature=0.0,max_output_tokens=4096)
                sql_query = get_sql_query_vertexai(question,llm)
           
            st.write('Generated SQL:')
            st.write(sql_query)
            df = run_on_bq(sql_query)
            st.write('DataFrame:', df)
            row_count = df.shape[0]
            column_count = df.shape[1]
            if row_count > 1 and column_count>1 :
                md_map = df.dtypes.to_dict()
                if llm_type == 'llama':
                    describe_prompt =f"""
                    <s>[INST] <<SYS>>
                    Assume you are a data analyst look into these context generated for my question which was '{question}'.
                    context :  values:{df}, metadeta:{md_map}
                    <</SYS>>
        
                     and give me a top 5 insights of this data in bullet points on seperate lines.
                    [/INST]
                    """
                else:
                    describe_prompt = f"""Assume you are a data analyst look into these values generated for my question which was '{question}' and give me a top 5 insights of this data in bullet points on seperate lines . values:{df}, metadeta:{md_map} """
                    
                description = describe_bot(describe_prompt, llm_type)
                st.write('Data Insights:', description)
                sample_prompts = sample_gen(df,llm_type)
                st.write('Sample Prompts:')
                st.write(sample_prompts) 
                
            if row_count>1 and column_count >1 and column_count < 3:
                generate_visualization(df)
                st.image('image_desc.png', caption='')        
                
        except Exception as e:
            main_placeholder.text(f"Unable to generate results . please correct the query.{e} !!")
            
 #-------------------------------------------------------------------------------          

if __name__ == '__main__':
    main()
