
import os

os.environ["HF_HOME"] = "C:/hf_cache"
os.environ["HF_HUB_CACHE"] = "C:/hf_cache/hub"

import time
from threading import Thread

import torch
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# ---------------------------
# Streamlit Configuration
# ---------------------------

st.set_page_config(
    page_title="Local AI Study Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Local AI Study Assistant")
st.caption("AI Llama Chatbot")

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SYSTEM_PROMPT = """
You are Qwen, an AI study assistant.

IDENTITY:
- You are an AI assistant, not a human.
- Your name is Qwen.
- The user's name is Murli.
- Never confuse your identity with the user's identity.
- Never claim to have a human career, personal life,
  experiences, hobbies or physical existence.

CONVERSATION RULES:
- Respond only to the latest user message.
- Use previous messages only when necessary for context.
- Never invent facts about the user.
- Never generate imaginary conversations.
- Never answer questions that the user did not ask.
- Never repeat your previous answer.
- Do not pretend to remember information that
  was never shared.

RESPONSE STYLE:
- Be natural, friendly and conversational.
- Use simple, understandable English.
- Keep casual answers within 1-3 sentences.
- For technical questions, explain step by step.
- Use examples when they genuinely help.
- Avoid unnecessary introductions.

IMPORTANT:
If asked about yourself, explain that you are
an AI assistant powered by a local language model.
If asked about the user's name, answer Murli.
"""

# ---------------------------
# Load Model Once
# ---------------------------

@st.cache_resource(show_spinner=False)
def load_model():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check your PyTorch installation."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir="C:/hf_cache/hub"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir="C:/hf_cache/hub",
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    model.to("cuda")
    model.eval()

    return tokenizer, model


with st.spinner("Loading model into GPU..."):
    tokenizer, model = load_model()

st.sidebar.success(
    f"GPU: {torch.cuda.get_device_name(0)}"
)

# ---------------------------
# Conversation Memory
# ---------------------------

if "memory" not in st.session_state:
    st.session_state.memory = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]

if st.sidebar.button("Clear Conversation"):
    st.session_state.memory = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]
    st.rerun()

# ---------------------------
# Generation Settings
# ---------------------------

max_tokens = st.sidebar.slider(
    "Maximum response tokens",
    min_value=64,
    max_value=512,
    value=160,
    step=32
)

# ---------------------------
# Display Conversation
# ---------------------------

for message in st.session_state.memory:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# ---------------------------
# Convert LangChain Messages
# ---------------------------

def prepare_messages(memory):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # Keep only the last 8 conversation messages
    for message in memory[1:][-8:]:

        if isinstance(message, HumanMessage):
            role = "user"

        elif isinstance(message, AIMessage):
            role = "assistant"

        else:
            continue

        messages.append({
            "role": role,
            "content": message.content
        })

    return messages

# ---------------------------
# Streaming Generator
# ---------------------------

def generate_response(memory, max_new_tokens):

    messages = prepare_messages(memory)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to("cuda")

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=120
    )

    generation_kwargs = {
    **inputs,
    "streamer": streamer,

    "max_new_tokens": max_tokens,

    "do_sample": False,

    "repetition_penalty": 1.05,

    "eos_token_id": tokenizer.eos_token_id,

    "pad_token_id": tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,

    "use_cache": True
}
    errors = []

    def run_generation():
        try:
            with torch.inference_mode():
                model.generate(**generation_kwargs)
        except Exception as exc:
            errors.append(exc)
            streamer.end()

    thread = Thread(
        target=run_generation,
        daemon=True
    )
    thread.start()

    for text in streamer:
        yield text

    thread.join()

    if errors:
        raise RuntimeError(
            f"Generation failed: {errors[0]}"
        )

# ---------------------------
# Chat Input
# ---------------------------

user_input = st.chat_input("Ask me anything...")

if user_input:

    st.session_state.memory.append(
        HumanMessage(content=user_input)
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        start_time = time.perf_counter()

        try:
            response = st.write_stream(
                generate_response(
                    st.session_state.memory,
                    max_tokens
                )
            )

            elapsed = time.perf_counter() - start_time

            st.caption(
                f"Generated in {elapsed:.2f} seconds"
            )

            st.session_state.memory.append(
                AIMessage(content=response)
            )

        except Exception as e:
            st.error(f"Error: {e}")

            # Remove failed user message
            st.session_state.memory.pop()
