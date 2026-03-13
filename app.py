import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import os
from datetime import datetime
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Center Matching System",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI Powered Center Matching System")
st.caption("Neural Matching Engine")

# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = SentenceTransformer("BAAI/bge-base-en")
    return model

model = load_model()

# --------------------------------------------------
# SYNONYM SYSTEM
# --------------------------------------------------

synonym_file = "synonyms.xlsx"

if os.path.exists(synonym_file):
    synonym_df = pd.read_excel(synonym_file)
else:
    synonym_df = pd.DataFrame(columns=["main_word","synonym"])

def build_synonym_dict(df):

    syn_dict = {}

    for _,row in df.iterrows():

        main = str(row["main_word"]).lower()
        syn = str(row["synonym"]).lower()

        if main not in syn_dict:
            syn_dict[main] = []

        syn_dict[main].append(syn)

    return syn_dict

synonyms = build_synonym_dict(synonym_df)

# --------------------------------------------------
# SIDEBAR SYNONYM EDITOR
# --------------------------------------------------

st.sidebar.header("🧠 Synonym Manager")

edited_synonyms = st.sidebar.data_editor(
    synonym_df,
    num_rows="dynamic",
    use_container_width=True
)

if st.sidebar.button("Save Synonyms"):

    edited_synonyms.to_excel(synonym_file,index=False)

    st.sidebar.success("Synonyms saved")

    synonyms = build_synonym_dict(edited_synonyms)

# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def normalize_synonyms(text):

    text = str(text).lower()

    for main_word, variations in synonyms.items():

        for word in variations:

            pattern = r"\b" + re.escape(word) + r"\b"

            text = re.sub(pattern, main_word, text)

    return text


def clean_text(text):

    text = str(text).lower()

    text = normalize_synonyms(text)

    text = re.sub(r"[^\w\s]","",text)

    text = re.sub(r"\s+"," ",text)

    return text.strip()

# --------------------------------------------------
# MEMORY SYSTEM
# --------------------------------------------------

memory_file = "learning_memory.csv"

if os.path.exists(memory_file):
    memory = pd.read_csv(memory_file)
else:
    memory = pd.DataFrame(columns=["input_text","match_center","master_id"])

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

col1,col2 = st.columns(2)

with col1:
    master_file = st.file_uploader("Upload Master File",type=["xlsx","csv"])

with col2:
    input_file = st.file_uploader("Upload Input File",type=["xlsx","csv"])

# --------------------------------------------------
# PROCESS FILES
# --------------------------------------------------

if master_file and input_file:

    if master_file.name.endswith(".csv"):
        master = pd.read_csv(master_file)
    else:
        master = pd.read_excel(master_file)

    if input_file.name.endswith(".csv"):
        input_data = pd.read_csv(input_file)
    else:
        input_data = pd.read_excel(input_file)

    st.success("Files Loaded")

# --------------------------------------------------
# CLEAN MASTER DATA
# --------------------------------------------------

    master["clean_name"] = master["center_name"].apply(clean_text)

    master["clean_address"] = master["address"].apply(clean_text)

    master["combined"] = (
        master["center_name"].astype(str) + " " +
        master["district"].astype(str) + " " +
        master["state"].astype(str) + " " +
        master["address"].astype(str)
    ).apply(clean_text)

# --------------------------------------------------
# GENERATE EMBEDDINGS
# --------------------------------------------------

    with st.spinner("Generating embeddings..."):

        embeddings = model.encode(master["combined"].tolist())

        embeddings = np.array(embeddings).astype("float32")

# --------------------------------------------------
# BUILD FAISS INDEX
# --------------------------------------------------

    dim = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dim,32)

    index.hnsw.efConstruction = 200

    index.add(embeddings)

# --------------------------------------------------
# MATCHING ENGINE
# --------------------------------------------------

    results=[]
    ids=[]
    scores=[]
    explanation=[]

    matched_address=[]
    matched_district=[]
    matched_state=[]

    progress = st.progress(0)

    total=len(input_data)

    for i,row in input_data.iterrows():

        progress.progress((i+1)/total)

        name = clean_text(row["center_name"])
        address = clean_text(row["address"])

        combined = f"{name} {row['district']} {row['state']} {address}"

# --------------------------------------------------
# MEMORY CHECK
# --------------------------------------------------

        mem = memory[memory["input_text"]==combined]

        if not mem.empty:

            master_row = master[master["center_id"]==mem.iloc[0]["master_id"]].iloc[0]

            results.append(mem.iloc[0]["match_center"])
            ids.append(mem.iloc[0]["master_id"])
            scores.append(1.0)
            explanation.append("Memory Recall")

            matched_address.append(master_row["address"])
            matched_district.append(master_row["district"])
            matched_state.append(master_row["state"])

            continue

# --------------------------------------------------
# VECTOR SEARCH
# --------------------------------------------------

        emb = model.encode([combined])
        emb = np.array(emb).astype("float32")

        k=5

        D,I = index.search(emb,k)

        candidates = master.iloc[I[0]]

# --------------------------------------------------
# FUZZY RANKING
# --------------------------------------------------

        best_score=0
        best=None
        best_reason=""

        for _,m in candidates.iterrows():

            name_score = fuzz.token_set_ratio(name,m["clean_name"])/100
            addr_score = fuzz.token_set_ratio(address,m["clean_address"])/100

            score=(0.6*name_score)+(0.4*addr_score)

            if score>best_score:

                best_score=score
                best=m
                best_reason=f"N:{name_score:.2f}|A:{addr_score:.2f}"

# --------------------------------------------------
# FINAL DECISION
# --------------------------------------------------

        if best_score>=0.90:

            results.append(best["center_name"])
            ids.append(best["center_id"])
            scores.append(best_score)
            explanation.append(best_reason)

            matched_address.append(best["address"])
            matched_district.append(best["district"])
            matched_state.append(best["state"])

        else:

            results.append("No Match")
            ids.append("NULL")
            scores.append(best_score)
            explanation.append(f"Low Confidence {best_score:.2f}")

            matched_address.append("")
            matched_district.append("")
            matched_state.append("")

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

    input_data["Matched Center"]=results
    input_data["Master ID"]=ids
    input_data["Matched Address"]=matched_address
    input_data["Matched District"]=matched_district
    input_data["Matched State"]=matched_state
    input_data["Score"]=scores
    input_data["Explanation"]=explanation

# --------------------------------------------------
# SHOW RESULTS
# --------------------------------------------------

    st.subheader("Results")

    st.dataframe(input_data,use_container_width=True)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

    match_rate=len([s for s in scores if s>0.9])/len(scores)*100
    avg_score=np.mean(scores)*100

    c1,c2,c3=st.columns(3)

    c1.metric("Match Rate",f"{match_rate:.1f}%")
    c2.metric("Average Confidence",f"{avg_score:.1f}%")
    c3.metric("Records",len(scores))

# --------------------------------------------------
# SAVE MEMORY
# --------------------------------------------------

    if st.button("Save Learning Memory"):

        for _,r in input_data.iterrows():

            if r["Matched Center"]!="No Match":

                txt=clean_text(
                    f"{r['center_name']} {r['district']} {r['state']} {r['address']}"
                )

                memory.loc[len(memory)]=[
                    txt,
                    r["Matched Center"],
                    r["Master ID"]
                ]

        memory.drop_duplicates(inplace=True)

        memory.to_csv(memory_file,index=False)

        st.success("Memory Updated")

# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------

    st.download_button(
        "Download Result CSV",
        input_data.to_csv(index=False),
        f"match_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )