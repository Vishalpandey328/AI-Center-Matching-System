import streamlit as st
import pandas as pd
import re
import os

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
st.title("AI Center Matching System")

# --------------------------
# Load NLP model
# --------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------
# Load synonym dictionary
# --------------------------
syn_file = "synonyms.xlsx"

if os.path.exists(syn_file):
    synonyms = pd.read_excel(syn_file)
else:
    synonyms = pd.DataFrame({
        "word":[
            "govt","rajkiya","mahila","balika","balak",
            "pg","inter","mahavidyalaya","vidyalaya","kanya"
        ],
        "replacement":[
            "government","government","girls","girls","boys",
            "postgraduate","intermediate","college","school","girls"
        ]
    })

# --------------------------
# Load reinforcement memory
# --------------------------
memory_file = "learning_memory.xlsx"

if os.path.exists(memory_file):
    memory = pd.read_excel(memory_file)

    required_cols = ["input_text","matched_center","master_id"]

    for col in required_cols:
        if col not in memory.columns:
            memory[col] = ""

    memory = memory[required_cols]

else:
    memory = pd.DataFrame(columns=["input_text","matched_center","master_id"])

# --------------------------
# Clean text + apply synonyms
# --------------------------
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^\w\s]", "", text)

    # apply synonyms
    for _,row in synonyms.iterrows():

        word = str(row["word"])
        repl = str(row["replacement"])

        text = re.sub(rf"\b{word}\b", repl, text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# --------------------------
# Hybrid matching function
# --------------------------
def hybrid_match(row, master):

    state = str(row["state"]).lower()
    district = str(row["district"]).lower()

    name = clean_text(row["center_name"])
    address = clean_text(row["address"])

    combined_input = f"{name} {district} {state} {address}"

    # reinforcement memory
    mem = memory[memory["input_text"] == combined_input]

    if not mem.empty:

        return mem.iloc[0]["matched_center"], mem.iloc[0]["master_id"], "NA","NA","NA",1.0

    subset = master[
        (master["state"].str.lower()==state) &
        (master["district"].str.lower()==district)
    ]

    if subset.empty:
        return "NA","NA","NA","NA","NA",0

    subset_embeddings = model.encode(subset["combined"].tolist())

    query_embedding = model.encode([combined_input])

    similarities = cosine_similarity(query_embedding,subset_embeddings)[0]

    best_score = 0
    best_row = None

    for i,(idx,row_master) in enumerate(subset.iterrows()):

        embed_score = similarities[i]

        fuzzy_name = fuzz.token_set_ratio(name,row_master["clean_name"])/100
        fuzzy_address = fuzz.token_set_ratio(address,row_master["clean_address"])/100

        final_score = (0.5*embed_score)+(0.3*fuzzy_name)+(0.2*fuzzy_address)

        if final_score > best_score:

            best_score = final_score
            best_row = row_master

    if best_score >= 0.90:

        return (
            best_row["center_name"],
            best_row["center_id"],
            best_row["district"],
            best_row["state"],
            best_row["address"],
            round(best_score,3)
        )

    else:
        return ("NA","NA","NA","NA","NA",round(best_score,3))

# --------------------------
# Upload files
# --------------------------
st.subheader("Upload Files")

col1,col2 = st.columns(2)

with col1:
    master_file = st.file_uploader("Upload Master Excel",type=["xlsx"])

with col2:
    input_file = st.file_uploader("Upload Input Excel",type=["xlsx"])

# --------------------------
# Processing
# --------------------------
if master_file and input_file:

    master = pd.read_excel(master_file)
    input_data = pd.read_excel(input_file)

    st.success("Files uploaded successfully")

    master["clean_name"] = master["center_name"].apply(clean_text)
    master["clean_address"] = master["address"].apply(clean_text)

    master["combined"] = (
        master["center_name"].astype(str)+" "+
        master["district"].astype(str)+" "+
        master["state"].astype(str)+" "+
        master["address"].astype(str)
    ).apply(clean_text)

    results=[]
    ids=[]
    mdistrict=[]
    mstate=[]
    maddress=[]
    scores=[]

    for _,row in input_data.iterrows():

        match,mid,district,state,address,score = hybrid_match(row,master)

        results.append(match)
        ids.append(mid)
        mdistrict.append(district)
        mstate.append(state)
        maddress.append(address)
        scores.append(score)

    input_data["Matched Center"]=results
    input_data["Master ID"]=ids
    input_data["Master District"]=mdistrict
    input_data["Master State"]=mstate
    input_data["Master Address"]=maddress
    input_data["Score"]=scores

    st.subheader("Matching Results")

    edited = st.data_editor(input_data,num_rows="dynamic")

# --------------------------
# Save feedback
# --------------------------
    if st.button("Save Feedback"):

        for _,r in edited.iterrows():

            if r["Matched Center"]!="NA":

                combined = clean_text(
                    f"{r['center_name']} {r['district']} {r['state']} {r['address']}"
                )

                memory.loc[len(memory)] = [
                    combined,
                    r["Matched Center"],
                    r["Master ID"]
                ]

        memory.drop_duplicates(inplace=True)

        memory.to_excel(memory_file,index=False)

        st.success("Feedback saved – reinforcement learning updated")

# --------------------------
# Add synonym UI
# --------------------------
st.subheader("Add Synonym")

col3,col4 = st.columns(2)

with col3:
    word = st.text_input("Word")

with col4:
    replacement = st.text_input("Replacement")

if st.button("Save Synonym"):

    synonyms.loc[len(synonyms)] = [word,replacement]

    synonyms.drop_duplicates(inplace=True)

    synonyms.to_excel(syn_file,index=False)

    st.success("Synonym saved successfully")

# --------------------------
# Download output
# --------------------------
if "edited" in locals():

    st.download_button(
        "Download Matched Data",
        edited.to_csv(index=False),
        "matched_output.csv"
    )