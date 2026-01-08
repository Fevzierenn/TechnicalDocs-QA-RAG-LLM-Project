import pandas as pd
import time
import os
import collections
import string

from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

# ==========================================
# 1. SETTINGS & API KEY
# ==========================================
# PASTE YOUR GROQ API KEY HERE
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# UPDATED: Llama 3.3 70B (Currently the strongest active model)
MODEL_NAME = "llama-3.3-70b-versatile"

VECTOR_DB_PATH = r"D:\LLM_PROJECT-DEEPLEARNING\vector_dbs\vector_db_smart_minilm"
COLLECTION_NAME = "langchain"
CSV_PATH = r"D:\LLM_PROJECT-DEEPLEARNING\golden-QA.csv"

PROMPT_TEMPLATE = """
You are an assistant that answers questions using the provided context.
CONTEXT:
{retrieved_chunks}
QUESTION:
{user_question}
INSTRUCTIONS:
- Use only the information in the context.
- If the answer is not in the context, say "I don't know based on the given documents."
- Answer concisely in English.
"""

# ==========================================
# 2. HELPER FUNCTION: GROQ REQUEST
# ==========================================
client = Groq(api_key=GROQ_API_KEY)

def ask_groq(prompt_text):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            model=MODEL_NAME,
            temperature=0.0,  # For consistent answers
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"\nGroq Error: {e}")
        # If rate limit is hit, wait a bit
        if "429" in str(e):
            print("Rate limit reached, waiting 10s...")
            time.sleep(10)
            return "RATE_LIMIT_ERROR"
        return f"ERROR: {str(e)}"

# ==========================================
# 3. METRIC FUNCTIONS
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

# ==========================================
# 4. TEST ENGINE
# ==========================================
def run_groq_evaluation():
    print(f"STARTING API TEST (GROQ - {MODEL_NAME})...")

    # 1. Load Data
    try:
        df = pd.read_csv(CSV_PATH, sep=';', encoding='cp1252')
        df.columns = df.columns.str.strip().str.lower()
        q_col = next((c for c in df.columns if 'question' in c))
        a_col = next((c for c in df.columns if 'answer' in c))
        test_data = df[[q_col, a_col]].to_dict('records')
        print(f"{len(test_data)} questions loaded.")
    except Exception as e:
        print(f"CSV Error: {e}")
        return

    # 2. Prepare Retriever
    print("Loading Vector DB...")
    ef = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=ef, collection_name=COLLECTION_NAME)

    # 3. Test Connection
    print(f"Testing Groq Connection...")
    try:
        test_cvp = ask_groq("Say 'Ready'")
        if "ERROR" in test_cvp:
             print(f"Connection error: {test_cvp}")
             return
        print(f"Connection PERFECT! (Response: {test_cvp})")
    except:
        print("Connection failed. Check API Key.")
        return

    all_logs = []
    csv_filename = f"RAG_API_GROQ2_{MODEL_NAME}_Results.csv"

    print(f"\nStarting Test...")

    for i, item in enumerate(tqdm(test_data)):
        question = item[q_col]
        ground_truth = item[a_col]

        # --- LEVEL B (NO RAG) ---
        start_b = time.time()
        res_b = ask_groq(question)
        end_b = time.time()

        f1_b = calculate_f1(res_b, ground_truth)

        # Groq is fast but wait 1-2 sec out of courtesy
        time.sleep(2)

        # --- LEVEL C (FULL RAG) ---
        try:
            docs = db.similarity_search(question, k=3)
            context_text = "\n".join([d.page_content for d in docs])
            full_prompt = PROMPT_TEMPLATE.format(retrieved_chunks=context_text, user_question=question)

            start_c = time.time()
            res_c = ask_groq(full_prompt)
            end_c = time.time()

            f1_c = calculate_f1(res_c, ground_truth)
            latency_c = end_c - start_c
        except Exception as e:
            print(f"RAG Error: {e}")
            res_c = "RAG_ERROR"
            f1_c, latency_c = 0, 0

        time.sleep(2)

        # Logging
        all_logs.append({
            "Model_Name": f"API-{MODEL_NAME}",
            "Question_ID": i + 1,
            "Question": question,
            "Ground_Truth": ground_truth,
            "Standard_LLM_Answer_(NO_RAG)": res_b,
            "Our_System_Answer_(WITH_RAG)": res_c,
            "Score_NO_RAG_(F1)": round(f1_b, 3),
            "Score_WITH_RAG_(F1)": round(f1_c, 3),
            "Success_Increase_(Diff)": round(f1_c - f1_b, 3),
            "Latency_(sec)": round(latency_c, 2),
            "Hardware_Usage": "CLOUD-API-GROQ",
            "Error_Status": "None" if "ERROR" not in str(res_c) else "Present"
        })

        if (i + 1) % 5 == 0:
            pd.DataFrame(all_logs).to_csv(csv_filename, index=False, sep=';', encoding='utf-8-sig')

    df_res = pd.DataFrame(all_logs)
    df_res.to_csv(csv_filename, index=False, sep=';', encoding='utf-8-sig')
    print(f"\nAPI Test Finished! Results: {csv_filename}")

if __name__ == "__main__":
    run_groq_evaluation()