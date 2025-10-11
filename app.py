import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv # <-- 1. IMPORTAR a biblioteca dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Carregar Variáveis de Ambiente ---
# Carrega as variáveis do arquivo .env para o ambiente do sistema
load_dotenv() 

# --- Configurações Iniciais da Página ---
st.set_page_config(
    page_title="Agente de Análise de Dados",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agente de Análise de Dados com Gemini")
st.markdown("""
Bem-vindo! Faça o upload de um arquivo CSV e comece a fazer perguntas sobre seus dados. 
O agente pode gerar descrições, análises e gráficos para você.
""")

# --- Obter a chave de API do ambiente ---
google_api_key = os.getenv("GOOGLE_API_KEY") # <-- 2. LER a chave do ambiente

# --- Sidebar para Upload ---
with st.sidebar:
    st.header("Configurações")
    # O input da chave de API foi REMOVIDO daqui para segurança.
    uploaded_file = st.file_uploader("Faça o upload do seu arquivo CSV", type=["csv"])

# --- Lógica principal da aplicação ---
# Verifica se a chave foi carregada e se um arquivo foi enviado
if not google_api_key:
    st.error("A chave de API do Google (GOOGLE_API_KEY) não foi encontrada.")
    st.info("Por favor, crie um arquivo .env e adicione sua chave de API nele.")
elif uploaded_file is None:
    st.info("Por favor, faça o upload de um arquivo CSV para começar.")
else:
    try:
        # Carregar os dados
        df = pd.read_csv(uploaded_file)
        st.success("Arquivo CSV carregado com sucesso! Veja uma amostra dos dados abaixo:")
        st.dataframe(df.head())

        # --- Inicialização do Agente ---
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", 
            google_api_key=google_api_key, # A chave agora vem da variável
            temperature=0.2
        )

        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True,
            agent_executor_kwargs={"handle_parsing_errors": True},
            allow_dangerous_code=True 
        )

        # --- Implementação da Memória e do Chat ---
        if "memory" not in st.session_state:
            st.session_state.memory = ""

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Faça sua pergunta sobre os dados..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            full_prompt = f"""
            Histórico da conversa anterior:
            {st.session_state.memory}

            Pergunta do usuário: {prompt}
            
            Instruções importantes: Responda em português do Brasil.
            """

            with st.chat_message("assistant"):
                with st.spinner("O agente está pensando... 🤔"):
                    try:
                        response = agent.invoke({"input": full_prompt})
                        st.markdown(response['output'])
                        
                        st.session_state.messages.append({"role": "assistant", "content": response['output']})
                        st.session_state.memory += f"\nUsuário: {prompt}\nAgente: {response['output']}"

                    except Exception as e:
                        st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")

    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o arquivo: {e}")