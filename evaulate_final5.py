import pandas as pd
import time
import psutil
import os
import collections
import string
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODELS_TO_TEST = ["qwen3:8b","gemma3:4b"]

VECTOR_DB_PATH = r"D:\LLM_PROJECT-DEEPLEARNING\vector_dbs\vector_db_smart_minilm"
COLLECTION_NAME = "langchain"
CSV_PATH = r"D:\LLM_PROJECT-DEEPLEARNING\golden-QA.csv"

# Prompt Template (Optimized for high F1 Score)
PROMPT_TEMPLATE = """
You are an assistant that answers questions using the provided context.
CONTEXT:
{retrieved_chunks}
QUESTION:
{user_question}
INSTRUCTIONS:
- Use only the information in the context.
-Use just the information inside the context, dont answer from models knowledge.
- If the answer is not in the context, say "I don't know based on the given documents."
- Answer concisely in English.
"""


# ==========================================
# 2. METRIC FUNCTIONS
# ==========================================
def normalize_text(s):
    if not s: return ""

    def white_space_fix(text): return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text): return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def calculate_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common_tokens = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def calculate_em(prediction, ground_truth):
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def get_hw_usage():
    return f"CPU:{psutil.cpu_percent()}% | RAM:{psutil.virtual_memory().percent}%"


# ==========================================
# 3. TEST ENGINE (SAFE MODE)
# ==========================================
def run_final_evaluation():
    print(f"[INFO] STARTING LONG-TERM TEST... ({len(MODELS_TO_TEST)} Models)")
    print("[INFO] Safety: Automatic save will occur after each model completes.")

    # 1. Load Data
    try:
        df = pd.read_csv(CSV_PATH, sep=';', encoding='cp1252')
        df.columns = df.columns.str.strip().str.lower()
        q_col = next((c for c in df.columns if 'question' in c))
        a_col = next((c for c in df.columns if 'answer' in c))
        test_data = df[[q_col, a_col]].to_dict('records')
        print(f"[OK] Total Questions: {len(test_data)}")
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        return

    # 2. Prepare Retriever
    print("[INFO] Loading database...")
    ef = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=ef, collection_name=COLLECTION_NAME)

    all_logs = []
    csv_filename = "RAG_Final_Results_Report2.csv"

    for model_name in MODELS_TO_TEST:
        print(f"\n[INFO] STARTING MODEL: {model_name.upper()}")
        print("-" * 60)

        try:
            llm = Ollama(model=model_name, temperature=0)
        except Exception as e:
            print(f"[WARNING] Model Loading Error ({model_name}): {e}")
            continue

        # Progress bar with TQDM
        for i, item in enumerate(tqdm(test_data, desc=f"{model_name} Test")):
            question = item[q_col]
            ground_truth = item[a_col]

            try:
                # --- LEVEL B (NO RAG) ---
                start_b = time.time()
                res_b = llm.invoke(question)
                end_b = time.time()

                f1_b = calculate_f1(res_b, ground_truth)
                em_b = calculate_em(res_b, ground_truth)
                latency_b = end_b - start_b

                # --- LEVEL C (FULL RAG) ---
                docs = db.similarity_search(question, k=3)
                context_text = "\n".join([d.page_content for d in docs])
                full_prompt = PROMPT_TEMPLATE.format(retrieved_chunks=context_text, user_question=question)

                start_c = time.time()
                res_c = llm.invoke(full_prompt)
                end_c = time.time()

                f1_c = calculate_f1(res_c, ground_truth)
                em_c = calculate_em(res_c, ground_truth)
                latency_c = end_c - start_c

                # Tokens per second approximation
                tps = (len(res_c.split()) * 1.3) / latency_c if latency_c > 0 else 0

                error_status = "None"

            except Exception as e:
                print(f"\n[WARNING] Question Error: {e}")
                res_b, res_c = "ERROR", "ERROR"
                f1_b, em_b, latency_b = 0, 0, 0
                f1_c, em_c, latency_c, tps = 0, 0, 0, 0
                error_status = str(e)

            # --- LOGGING (ENGLISH COLUMN NAMES) ---
            all_logs.append({
                "Model_Name": model_name,
                "Question_ID": i + 1,
                "Question": question,
                "Ground_Truth": ground_truth,

                # RESPONSE COLUMNS:
                "Standard_LLM_Answer_(NO_RAG)": res_b,
                "Our_System_Answer_(WITH_RAG)": res_c,

                # SCORES:
                "Score_NO_RAG_(F1)": round(f1_b, 3),
                "Score_WITH_RAG_(F1)": round(f1_c, 3),
                "Success_Increase_(Diff)": round(f1_c - f1_b, 3),

                # METRICS:
                "Latency_(sec)": round(latency_c, 2),
                "Hardware_Usage": get_hw_usage(),
                "Error_Status": error_status,

                # HUMAN EVALUATION (PLACEHOLDERS):
                "Human_Relevance(1-5)": "",
                "Human_Accuracy(1-5)": "",
                "Hallucination(Yes/No)": ""
            })

        # --- INTERMEDIATE SAVE (AFTER EACH MODEL) ---
        try:
            df_temp = pd.DataFrame(all_logs)
            df_temp.to_csv(csv_filename, index=False, sep=';', encoding='utf-8-sig')
            print(f"[SAVED] {model_name} completed, data written to '{csv_filename}'.")
        except Exception as e:
            print(f"[ERROR] File write error (Is the file open?): {e}")

    print(f"\n[DONE] ALL TESTS FINISHED! Results are in '{csv_filename}'.")


if __name__ == "__main__":
    run_final_evaluation()