import requests
import json
import streamlit as st

# Função principal para enviar perguntas ao modelo no Groq
def ask_ai(prompt, contexto=None):
    # treinamento
    system = "Você é um analista financeiro e assistente do app. Responda de forma clara e objetiva."

    # Anexa contexto 
    if contexto:
        system += "\nContexto: " + json.dumps(contexto, ensure_ascii=False)

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {st.secrets['groq_api_key']}"
    }

    payload = {
        "model": "qwen/qwen3-32b",  
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2  
    }

    # Chamada à API
    resp = requests.post(url, headers=headers, json=payload)
    data = resp.json()

    # Retorna a resposta textual
    return data["choices"][0]["message"]["content"]
