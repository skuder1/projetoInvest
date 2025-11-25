import requests
import streamlit as st

def ask_ai(prompt):

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # treinamento
    system = (
        "Você é um analista financeiro do aplicativo do usuário. "
        "Explique preços, movimentos de mercado e responda de forma clara. "
        'Responda sempre na linguagem da pergunta do usuário'
    )

    messages = [{"role": "system", "content": system}]

    for item in st.session_state.chat_history:
        messages.append(item)

    messages.append({"role": "user", "content": prompt})

    # Chamada à API
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {st.secrets['groq_api_key']}"
    }
    payload = {
        "model": "qwen/qwen3-32b",
        "messages": messages,
        "temperature": 0.2
    }

    resp = requests.post(url, headers=headers, json=payload)
    data = resp.json()
    resposta = data["choices"][0]["message"]["content"]

    # salva no histórico
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": resposta})

    return resposta
