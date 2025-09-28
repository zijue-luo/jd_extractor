# app_web.py
import os
import streamlit as st
from dotenv import load_dotenv

from core.logic import JDWorker, Configurations

load_dotenv()

st.set_page_config(page_title="JD Skills Extractor - Minimal GUI", layout="wide")
st.title("JD Skills Extractor (Minimal GUI)")

# --- Central place to edit model options ---
MODEL_OPTIONS = {
    "openai": [
        "gpt-5",
        "gpt-4o",
        "Custom...",  # keep this at the end
    ],
    "gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "Custom...",  # keep this at the end
    ],
}

# --- Keep one worker instance in session ---
if "worker" not in st.session_state:
    st.session_state.worker = JDWorker(Configurations())

worker: JDWorker = st.session_state.worker

# --- Sidebar: model settings & API keys ---
with st.sidebar:
    st.header("Model Settings")

    provider = st.selectbox("Provider", ["openai", "gemini"], index=0)
    worker.set_provider(provider)

    # model dropdown sourced from MODEL_OPTIONS
    choices = MODEL_OPTIONS.get(provider, [])
    # try to select current model if it exists in choices; else default to first
    default_index = choices.index(worker.config.model) if worker.config.model in choices else 0
    picked = st.selectbox("Model", choices, index=default_index)

    if picked == "Custom...":
        custom_model = st.text_input("Custom model name", value=worker.config.model if worker.config.model not in choices else "")
        model_to_use = custom_model.strip() or choices[0]
    else:
        model_to_use = picked

    worker.set_model(model_to_use)

    temperature = st.slider("Temperature", 0.0, 1.0, worker.config.temperature, 0.1)
    worker.set_temperature(temperature)

    reasoning = st.selectbox(
        "Reasoning effort",
        ["low", "medium", "high", "minimal", "dynamic"],
        index=["low", "medium", "high", "minimal", "dynamic"].index(worker.config.reasoning_effort),
    )
    worker.set_reasoning(reasoning)

    sys_prompt = st.text_area("System prompt (optional)", value=worker.config.system_prompt, height=140)
    worker.set_system_prompt(sys_prompt)

    st.markdown("---")
    st.header("API Keys")

    openai_env = os.environ.get("OPENAI_API_KEY") or ""
    gemini_env = os.environ.get("GEMINI_API_KEY") or ""

    openai_key_input = st.text_input(
        "OPENAI_API_KEY (optional, overrides env if set)",
        value="",
        type="password",
        help="Leave empty to use environment variable",
    )
    gemini_key_input = st.text_input(
        "GEMINI_API_KEY (optional, overrides env if set)",
        value="",
        type="password",
        help="Leave empty to use environment variable",
    )

    if provider == "openai":
        worker.set_api_key("openai", openai_key_input or openai_env or None)
    else:
        worker.set_api_key("gemini", gemini_key_input or gemini_env or None)

# --- Main: JD input -> Generate -> Edit -> Save ---
st.subheader("1) Paste Job Description")
jd_text = st.text_area("Job Description", height=260, placeholder="Paste JD text here...")

col_gen, col_save = st.columns(2)

with col_gen:
    st.subheader("2) Generate JSON")
    if st.button("Generate", type="primary", use_container_width=True, disabled=not jd_text.strip()):
        try:
            ai_text = worker.generate(jd_text)
            st.session_state["ai_text"] = ai_text
            st.success("Generated. You can edit the JSON below before saving.")
        except Exception as e:
            st.error(f"Generation failed: {e}")

st.subheader("3) Review / Edit JSON (must be valid JSON)")
ai_text_default = st.session_state.get("ai_text", "")
ai_text_box = st.text_area("LLM Output (editable)", value=ai_text_default, height=300)

with col_save:
    st.subheader("4) Save")
    save_path = st.text_input("JSONL Path", value="data/extracted_skills.jsonl")
    if st.button("Save to JSONL", use_container_width=True, disabled=not (ai_text_box.strip() and save_path.strip())):
        try:
            worker.save(ai_text_box, save_path)
            st.success(f"Saved to: {save_path}")
        except Exception as e:
            st.error(f"Save failed: {e}")

st.caption("Notes: Use the sidebar to switch provider and model. Edit MODEL_OPTIONS at the top to change dropdown choices.")
