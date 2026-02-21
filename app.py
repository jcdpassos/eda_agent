import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Carregar Variáveis de Ambiente (Apenas uma vez no início)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- Configurações Iniciais da Página ---
st.set_page_config(
    page_title="Agente de Análise de Dados",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agente de Análise Exploratóriade Dados")
st.markdown("""
Eu sou um agente de IA especializado em análise e visualização de dados. 
Faça o upload de um arquivo CSV e me peça para gerar insights e gráficos.
""")

# --- Sidebar para Upload ---
with st.sidebar:
    st.header("Configurações")
    uploaded_file = st.file_uploader("Faça o upload do seu arquivo CSV", type=["csv"])

# --- Lógica principal da aplicação ---
if not api_key:
    st.error("A chave de API (GOOGLE_API_KEY) não foi encontrada no arquivo .env.")
elif uploaded_file is None:
    st.info("Por favor, faça o upload de um arquivo CSV para começar.")
else:
    try:
        # Carregamento dos dados
        df = pd.read_csv(uploaded_file)
        
        st.success("Arquivo CSV carregado com sucesso!")
        with st.expander("Ver amostra dos dados"):
            st.dataframe(df.head())
        
        # --- Inicialização do Agente ---
        # Usando o modelo que você definiu (gemini-2.0-flash)
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1
)
        agent_executor = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True,
            agent_executor_kwargs={"handle_parsing_errors": True},
            allow_dangerous_code=True 
        )

        # --- Gerenciamento do Estado do Chat ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Exibe histórico
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if "plot" in message:
                    st.pyplot(message["plot"])
                else:
                    st.markdown(message["content"])

        # Input do Usuário
        if prompt := st.chat_input("Ex: 'Qual a média da coluna X?' ou 'Crie um gráfico de barras...'"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analisando..."):
                    # Prompt robusto para o agente
                    full_prompt = f"""
                    Você é um especialista em Python/Pandas. Responda à pergunta sobre o dataframe 'df'.
                    Pergunta: {prompt}
                    
                    REGRAS PARA GRÁFICOS:
                    - Se pedir gráfico, use Matplotlib.
                    - Defina a figura como 'fig, ax = plt.subplots()'.
                    - A última linha deve ser apenas a variável 'fig'.
                    - NÃO use plt.show().
                    - Responda em Português (Brasil).
                    """

                    try:
                        response = agent_executor.invoke({"input": full_prompt})
                        output_text = response.get('output', '')

                        # Verifica se o agente gerou uma figura no Matplotlib
                        fig = plt.gcf()
                        
                        if len(fig.get_axes()) > 0:
                            st.pyplot(fig)
                            st.session_state.messages.append({"role": "assistant", "plot": fig, "content": "Gráfico gerado."})
                            plt.close(fig) # Importante para liberar memória
                        else:
                            st.markdown(output_text)
                            st.session_state.messages.append({"role": "assistant", "content": output_text})
                        
                        plt.clf() 

                    except Exception as e:
                        st.error(f"Erro no processamento: {e}")

    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")