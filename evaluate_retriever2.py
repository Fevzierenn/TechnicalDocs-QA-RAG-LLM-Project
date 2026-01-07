import os
import csv
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# 1. SETTINGS
# ==========================================
print("--- PHASE 2: RETRIEVER PERFORMANCE TEST (FIXED) ---")

script_dir = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(script_dir, "vector_dbs")

# Full name of the CSV file
EVAL_SET_PATH = os.path.join(script_dir, "golden-QA.csv")

# K values to test
K_VALUES = [1, 3, 4, 5]

# Model Definitions
EMBEDDING_MODELS_MAP = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5"
}


# ==========================================
# 2. ENHANCED CSV READER
# ==========================================
def load_eval_set_from_csv():
    if not os.path.exists(EVAL_SET_PATH):
        print(f"ERROR: '{EVAL_SET_PATH}' not found! Check the file name.")
        exit()

    data = []
    # Excel usually uses cp1252, Python likes utf-8. Let's try them all.
    encodings_to_try = ['utf-8-sig', 'cp1252', 'latin-1']

    file_content = None
    used_encoding = None

    # 1. Find the Correct Encoding
    for enc in encodings_to_try:
        try:
            with open(EVAL_SET_PATH, 'r', encoding=enc) as f:
                file_content = f.readlines()
            used_encoding = enc
            print(f"Reading CSV ({enc}): {EVAL_SET_PATH}")
            break
        except UnicodeDecodeError:
            continue

    if not file_content:
        print("ERROR: The file could not be read in any format!")
        exit()

    # 2. Find Delimiter (Comma or Semicolon?)
    # It looks like ';' in your data, let's try that first.
    delimiter = ','
    if ';' in file_content[0]:
        delimiter = ';'

    print(f"Detected delimiter: '{delimiter}'")

    # 3. Parse Content
    reader = csv.reader(file_content, delimiter=delimiter)

    # Skip header (QUESTIONS;ANSWERS;PATH)
    header = next(reader, None)

    for row in reader:
        if len(row) < 3:
            continue  # Skip if row is incomplete

        # Your CSV format: QUESTIONS;ANSWERS;PATH
        question = row[0].strip()
        source_doc = row[2].strip()  # PATH is in the 3rd column (index 2)

        if question and source_doc:
            data.append({
                "question": question,
                "source_doc": source_doc
            })

    print(f"Loaded {len(data)} question-answer pairs from CSV.")
    return data


# ==========================================
# 3. EVALUATION ENGINE
# ==========================================
def evaluate_db(db_path, eval_data, embed_model_name):
    db_basename = os.path.basename(db_path)
    print(f"\nTesting: {db_basename}")

    # Load DB and Model
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)

    results = {k: {"hits": 0, "recall": 0.0} for k in K_VALUES}
    total_questions = len(eval_data)

    start_time = time.time()

    for item in eval_data:
        question = item['question']
        target_source = item['source_doc']

        # File extension cleanup (path vs name)
        target_clean = os.path.basename(target_source)

        # Retrieve up to the largest k value
        max_k = max(K_VALUES)
        retrieved_docs = db.similarity_search(question, k=max_k)

        # List retrieved source names
        retrieved_sources = [os.path.basename(doc.metadata.get('source', '')) for doc in retrieved_docs]

        for k in K_VALUES:
            current_retrieved = retrieved_sources[:k]

            # Match Check
            match_found = False
            for retrieved_doc in current_retrieved:
                # Sometimes .md is missing, sometimes full path comes.
                # Checking both ways.
                if target_clean in retrieved_doc or retrieved_doc in target_clean:
                    match_found = True
                    break

            if match_found:
                results[k]["hits"] += 1

    duration = time.time() - start_time

    # Print Results
    print(f"   Question Count: {total_questions} | Duration: {duration:.2f}s")
    for k in K_VALUES:
        hit_rate = results[k]["hits"] / total_questions
        print(f"   Recall@{k}: {hit_rate:.2%}")

    return results


# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    # Load CSV (Should not error now)
    eval_data = load_eval_set_from_csv()

    if not os.path.exists(VECTOR_DB_DIR):
        print("Database folder does not exist. Run create_db.py first.")
        exit()

    db_folders = [f for f in os.listdir(VECTOR_DB_DIR) if f.startswith("vector_db_")]

    for folder_name in db_folders:
        full_path = os.path.join(VECTOR_DB_DIR, folder_name)

        # Infer model from folder name
        if "minilm" in folder_name:
            model_key = EMBEDDING_MODELS_MAP["minilm"]
        elif "bge" in folder_name:
            model_key = EMBEDDING_MODELS_MAP["bge"]
        else:
            continue

        evaluate_db(full_path, eval_data, model_key)

    print("\n--- TEST COMPLETED ---")