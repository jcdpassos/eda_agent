

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt # Importa Matplotlib
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Carregar Variáveis de Ambiente
load_dotenv()

# --- Configurações Iniciais da Página ---
st.set_page_config(
    page_title="Agente de Análise de Dados",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agente Analista de Dados com Gemini")
st.markdown("""
Eu sou um agente de IA especializado em análise e visualização de dados. 
Faça o upload de um arquivo CSV e me peça para gerar insights, resumos e gráficos diversificados.
""")

# --- Obter a chave de API do ambiente ---
google_api_key = os.getenv("GOOGLE_API_KEY")

# --- Sidebar para Upload ---
with st.sidebar:
    st.header("Configurações")
    uploaded_file = st.file_uploader("Faça o upload do seu arquivo CSV", type=["csv"])

# --- Lógica principal da aplicação ---
if not google_api_key:
    st.error("A chave de API do Google (GOOGLE_API_KEY) não foi encontrada.")
    st.info("Por favor, crie um arquivo .env e adicione sua chave de API nele.")
elif uploaded_file is None:
    st.info("Por favor, faça o upload de um arquivo CSV para começar.")
else:
    try:
        df = pd.read_csv(uploaded_file)
        
        # <<< INÍCIO DA ALTERAÇÃO SOLICITADA >>>
        # Adiciona uma mensagem de sucesso e exibe as primeiras linhas do dataframe.
        st.success("Arquivo CSV carregado com sucesso!")
        st.subheader("Amostra dos Dados:")
        # O comando st.dataframe() cria uma tabela interativa na tela.
        # Usamos df.head() para mostrar apenas as 5 primeiras linhas,
        # o que evita sobrecarregar a tela com arquivos muito grandes.
        st.dataframe(df.head())
        # <<< FIM DA ALTERAÇÃO SOLICITADA >>>
        
        # --- Inicialização do Agente ---
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", 
            google_api_key=google_api_key,
            temperature=0.1 # Temperatura mais baixa para respostas mais precisas
        )

        # O 'agent_executor' é o que realmente executa as tarefas
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

        # Exibe mensagens do histórico a cada nova execução do script
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # MELHORIA 1: Verifica se a mensagem é um gráfico ou texto
                if "plot" in message:
                    st.pyplot(message["plot"])
                else:
                    st.markdown(message["content"])

        # Captura a pergunta do usuário
        if prompt := st.chat_input("Ex: 'Crie um mapa de calor de correlações'"):
            # Adiciona a pergunta do usuário ao histórico para exibição
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Prepara a mensagem para o assistente
            with st.chat_message("assistant"):
                with st.spinner("Analisando e gerando resposta... 🤔"):
                    # MELHORIA 2: Prompt de Engenharia Detalhado para Gráficos
                    full_prompt = f"""
                    Você é um agente de análise de dados especialista em Python, Pandas, Matplotlib e Seaborn.

                    Sua tarefa é responder à pergunta do usuário sobre o dataframe fornecido.

                    PERGUNTA DO USUÁRIO: "{prompt}"

                    INSTRUÇÕES PARA GERAR GRÁFICOS:
                    1. QUANDO uma pergunta exigir um gráfico, GERE o código para criar a visualização usando Matplotlib ou Seaborn.
                    2. **NUNCA use 'plt.show()'**. Em vez disso, a última linha do seu bloco de código de plotagem deve ser a figura em si (ex: `fig`). 
                    3. Se você criar uma figura com `plt.subplots()`, certifique-se de que a variável `fig` seja retornada no final.
                    4. Use títulos claros, rótulos de eixos e legendas para tornar o gráfico informativo.
                    5. Se a pergunta for ambígua, faça uma suposição razoável sobre o tipo de gráfico a ser gerado.
                    6. Responda sempre em português do Brasil.

                    Exemplo de como gerar um gráfico de dispersão:
                    ```python
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    ax.scatter(df['coluna_x'], df['coluna_y'])
                    ax.set_title('Título do Gráfico')
                    ax.set_xlabel('Eixo X')
                    ax.set_ylabel('Eixo Y')
                    # A última linha é a variável 'fig', não 'plt.show()'
                    fig
                    ```
                    
                    Agora, analise a pergunta do usuário e forneça a resposta em texto ou gere o código para o gráfico.
                    """

                    try:
                        # Executa o agente com o prompt detalhado
                        response = agent_executor.invoke({"input": full_prompt})
                        
                        # MELHORIA 3: Lógica para tratar a saída (texto ou gráfico)
                        output_text = response.get('output', 'Não foi possível gerar uma resposta.')
                        
                        # O LangChain experimentalmente coloca o gráfico em 'stdout' ou similar.
                        # Uma maneira mais robusta é verificar se uma figura foi criada.
                        # A abordagem mais simples é renderizar o texto e verificar a figura global.
                        
                        fig = plt.gcf() # Pega a figura global corrente do Matplotlib
                        
                        # Se a figura tiver eixos (ou seja, não está vazia), ela é um gráfico
                        if fig.get_axes():
                            st.pyplot(fig)
                            st.session_state.messages.append({"role": "assistant", "plot": fig})
                        else:
                            st.markdown(output_text)
                            st.session_state.messages.append({"role": "assistant", "content": output_text})
                        
                        plt.clf() # Limpa a figura para o próximo gráfico

                    except Exception as e:
                        st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")

    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o arquivo: {e}")