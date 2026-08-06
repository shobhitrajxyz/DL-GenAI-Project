import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack

st.set_page_config(page_title="Smart MCQ Solver AI", page_icon="🧠", layout="centered")

st.title("Smart MCQ Solver AI")
st.markdown("**Shobhit Raj (Roll No: 24f2008744)** — *Model Deployment*")
st.write("Enter a question prompt and 5 options to get real-time AI option rankings!")

# Load model artifacts
@st.cache_resource
def load_model():
    return joblib.load("mcq_solver_model.joblib")

try:
    artifacts = load_model()
    m4_lr = artifacts['m4_lr']
    tfidf_word = artifacts['tfidf_word']
    scaler = artifacts['scaler']
    model_loaded = True
except Exception as e:
    st.warning("⚠️ Model file `mcq_solver_model.joblib` not found. Please upload it alongside this app.")
    model_loaded = False

def extract_tabular_features_single(prompt, options_dict):
    prompt_words = set(str(prompt).lower().split())
    lens = [len(str(options_dict[c])) for c in ['A', 'B', 'C', 'D', 'E']]
    mean_len = np.mean(lens) + 1e-5
    max_len = np.max(lens)
    min_len = np.min(lens)
    
    rows = []
    for c in ['A', 'B', 'C', 'D', 'E']:
        opt = str(options_dict[c])
        char_len = len(opt)
        words = opt.split()
        word_len = len(words)
        mean_w_len = char_len / (word_len + 1e-5)
        len_diff = char_len - mean_len
        len_ratio = char_len / mean_len
        is_longest = 1.0 if char_len == max_len else 0.0
        is_shortest = 1.0 if char_len == min_len else 0.0
        punc_cnt = sum(1 for ch in opt if ch in ",.;:-()!?\'\"")
        cap_cnt = sum(1 for ch in opt if ch.isupper())
        dig_cnt = sum(1 for ch in opt if ch.isdigit())
        opt_words = set(opt.lower().split())
        overlap = len(prompt_words.intersection(opt_words)) / (len(prompt_words) + 1e-5)
        
        rows.append([
            char_len, word_len, mean_w_len, len_diff, len_ratio,
            is_longest, is_shortest, punc_cnt, cap_cnt, dig_cnt, overlap
        ])
    return np.array(rows)

with st.form("mcq_form"):
    prompt = st.text_area("Question / Prompt", value="Which technique is commonly used for dimensionality reduction in machine learning?", height=90)
    
    col1, col2 = st.columns(2)
    with col1:
        opt_a = st.text_input("Option A", value="Principal Component Analysis (PCA)")
        opt_b = st.text_input("Option B", value="Linear Regression")
        opt_c = st.text_input("Option C", value="K-Means Clustering")
    with col2:
        opt_d = st.text_input("Option D", value="Decision Trees")
        opt_e = st.text_input("Option E", value="Gradient Boosting")

    submit = st.form_submit_button("🚀 Solve MCQ")

if submit:
    if not model_loaded:
        st.error("Model is not loaded. Upload `mcq_solver_model.joblib` to activate real predictions.")
    else:
        options = {'A': opt_a, 'B': opt_b, 'C': opt_c, 'D': opt_d, 'E': opt_e}
        
        # Extract features
        raw_tab = extract_tabular_features_single(prompt, options)
        norm_tab = scaler.transform(raw_tab)
        
        texts = [f"{prompt} [SEP] {options[c]}" for c in ['A', 'B', 'C', 'D', 'E']]
        text_feats = tfidf_word.transform(texts)
        all_feats = hstack([text_feats, norm_tab])
        
        # Get probabilities
        probs = m4_lr.predict_proba(all_feats)[:, 1]
        exp_p = np.exp(probs - np.max(probs))
        norm_probs = exp_p / np.sum(exp_p)
        
        sorted_indices = np.argsort(norm_probs)[::-1]
        opt_chars = ['A', 'B', 'C', 'D', 'E']
        
        best_opt = opt_chars[sorted_indices[0]]
        best_text = options[best_opt]
        best_conf = norm_probs[sorted_indices[0]] * 100
        
        st.success(f"### 🏆 Best Answer: Option {best_opt} ({best_conf:.1f}% confidence)\n**{best_text}**")
        
        st.subheader("📊 Ranking Breakdown:")
        for rank, idx in enumerate(sorted_indices):
            char = opt_chars[idx]
            text = options[char]
            pct = norm_probs[idx] * 100
            st.write(f"**Rank {rank+1}: Option {char}** ({pct:.1f}%) — {text}")
            st.progress(float(norm_probs[idx]))
