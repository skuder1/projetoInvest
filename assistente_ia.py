import requests
import streamlit as st

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def ask_ai(prompt):

    # cria histórico se não existir
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Prompt anti-chain-of-thought
    system_msg = (
        "Você é um analista financeiro com acesso à internet em tempo real."
        "Use navegação web quando necessário para buscar informações atuais, traga movimentações de mercado, contexto, etc."
        "Responda sempre na linguagem do usuário."
        "Forneça apenas conclusões claras e úteis."
    )

    # histórico
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(st.session_state.chat_history)
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {st.secrets['openrouter_api_key']}",
        "HTTP-Referer": "https://projetoliftup-wshhm7rv3yb26cyrf9wn7e.streamlit.app/", 
        "X-Title": "LiftUp Finance AI",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "x-ai/grok-4.1-fast:free",
        "messages": messages,
        "temperature": 0.2
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=40)
        resp.raise_for_status()
        data = resp.json()
        resposta = data["choices"][0]["message"]["content"]
    except Exception as e:
        resposta = f"Erro ao consultar IA: {e}"

    # salva histórico
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": resposta})

    return resposta


def clear_history():
    st.session_state.chat_history = []
