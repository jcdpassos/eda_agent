import google.generativeai as genai
import os
from dotenv import load_dotenv

# Carrega a chave de API do arquivo .env
load_dotenv()

print("Verificando modelos disponíveis...")

try:
    # Configura a API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Erro: A variável GOOGLE_API_KEY não foi encontrada no arquivo .env.")
    else:
        genai.configure(api_key=api_key)

        print("\nModelos disponíveis que suportam o método 'generateContent':")
        # Lista os modelos
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)

except Exception as e:
    print(f"\nOcorreu um erro ao conectar-se à API do Google: {e}")