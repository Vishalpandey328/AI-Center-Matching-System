import pandas as pd
import os

from preprocessing import clean_text
from embeddings import create_embeddings
from vector_store import build_index, search_index


print("Program started")

# Load master data
master = pd.read_excel("data/master_centers.xlsx")

# Load input data
input_data = pd.read_excel("data/input_centers.xlsx")

print("Files loaded")

# Clean text
master["clean_name"] = master["center_name"].apply(clean_text)
input_data["clean_name"] = input_data["center_name"].apply(clean_text)

print("Preprocessing completed")

# Create embeddings for master centers
master_embeddings = create_embeddings(master["clean_name"].tolist())

print("Embeddings created")

# Build FAISS index
index = build_index(master_embeddings)

print("Vector index built")

# Load learning memory
memory_file = "learning_memory.xlsx"

if os.path.exists(memory_file):

    memory = pd.read_excel(memory_file)

else:

    memory = pd.DataFrame(columns=["input_name", "matched_center"])


results = []

print("Starting matching")

for i, row in input_data.iterrows():

    input_name = row["clean_name"]

    original_name = row["center_name"]

    # Check reinforcement memory first
    memory_match = memory[memory["input_name"] == input_name]

    if not memory_match.empty:

        best_match = memory_match.iloc[0]["matched_center"]

    else:

        query_embedding = create_embeddings([input_name])

        distances, indices = search_index(index, query_embedding)

        best_index = indices[0][0]

        best_match = master.iloc[best_index]["center_name"]

        # Store reinforcement learning
        memory.loc[len(memory)] = [input_name, best_match]

    results.append({
        "Input Center": original_name,
        "Matched Center": best_match
    })

    print(f"{original_name} → {best_match}")

# Save output
output = pd.DataFrame(results)

output.to_excel("matched_output.xlsx", index=False)

# Save reinforcement memory
memory.to_excel(memory_file, index=False)

print("Matching completed")
print("Output saved to matched_output.xlsx")
print("Learning memory updated")