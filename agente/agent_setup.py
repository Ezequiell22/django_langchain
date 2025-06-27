import os
from dotenv import load_dotenv, find_dotenv
from urllib.parse import quote_plus

from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.agents.agent_types import AgentType
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

load_dotenv(find_dotenv())

odbc_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={os.getenv('SERVER_DB')};"
    f"DATABASE={os.getenv('DATABASE')};"
    f"UID={os.getenv('USER_DB')};"
    f"PWD={os.getenv('PASS_DB')};"
    f"TrustServerCertificate=yes;"
)
quoted = quote_plus(odbc_str)
connection_string = f"mssql+pyodbc:///?odbc_connect={quoted}"

db = SQLDatabase.from_uri(
    connection_string,
    include_tables=["SE1010", "SB1010", "SA1010", "SD1010", "SF2010", "SF1010", "SE2010"],
    engine_args={"echo": False},
    sample_rows_in_table_info=0
)

llm = ChatOpenAI(temperature=0, model="gpt-4o")

sql_agent = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=False
)

schema_docs = """
Tabela SE1010: Cadastro de transportadoras com dados como nome, CNPJ, endereço e código de identificação.
Tabela SB1010: Cadastro de produtos com informações como descrição, unidade, preço e código de barras.
Tabela SA1010: Cadastro de clientes com dados como nome, endereço, CPF/CNPJ, e cidade.
Tabela SD1010: Cadastro de fornecedores com dados fiscais e de contato.
Tabela SF1010: Cabeçalho das notas fiscais, contendo data, número da nota, cliente e valores totais.
Tabela SF2010: Itens das notas fiscais, com produto, quantidade, valor unitário e total.
Tabela SE2010: Conhecimentos de transporte, com dados sobre o frete, transportadora e destinatário.
"""
documents = CharacterTextSplitter(chunk_size=1000).split_documents([Document(page_content=schema_docs)])
vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())

analista_prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um analista financeiro. Analise os dados abaixo e gere um relatório textual "
               "Use linguagem profissional. Identifique totais, agrupamentos, padrões ou anomalias."
               "Seja breve e monte uma resposta resumida."),
    ("user", "Pergunta original: {pergunta}\n\nDados retornados:\n{dados}")
])

analista_financeiro_chain = LLMChain(
    llm=llm,
    prompt=analista_prompt,
    verbose=False
)
