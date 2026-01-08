import pandas as pd
import time
import psutil
import os
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rouge_score import rouge_scorer
from tqdm import tqdm

# --- AYARLAR ---
# Qwen çok yavaş olduğu için istersen listeden çıkarabilirsin:
MODELS_TO_TEST = [ "gemma3:4b","phi3"]

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
- Answer concisely in Turkish.
"""


def calculate_score(prediction, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    if not prediction: prediction = ""
    scores = scorer.score(ground_truth, prediction)
    return scores['rougeL'].fmeasure


def get_hw_usage():
    return f"CPU:{psutil.cpu_percent()}% | RAM:{psutil.virtual_memory().percent}%"


def run_evaluation():
    print(f"🚀 LOKAL LLM TESTİ (GÜVENLİ MOD) BAŞLIYOR... ({len(MODELS_TO_TEST)} Model)")

    try:
        df = pd.read_csv(CSV_PATH, sep=';', encoding='cp1252')
        df.columns = df.columns.str.strip().str.lower()
        q_col = next((c for c in df.columns if 'question' in c))
        a_col = next((c for c in df.columns if 'answer' in c))
        # Hız kazanmak istersen burayı .head(3) yapabilirsin
        test_data = df[[q_col, a_col]].head(3).to_dict('records')
    except Exception as e:
        print(f"❌ CSV okunamadı: {e}")
        return

    print("⏳ Veritabanı yükleniyor...")
    ef = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=ef, collection_name=COLLECTION_NAME)

    results = []

    for model_name in MODELS_TO_TEST:
        print(f"\n🤖 MODEL TEST EDİLİYOR: {model_name.upper()}")
        print("-" * 50)

        try:
            llm = Ollama(model=model_name, temperature=0)
        except Exception as e:
            print(f"⚠️ Model hatası: {e}")
            continue

        for i, item in enumerate(tqdm(test_data)):
            question = item[q_col]
            ground_truth = item[a_col]

            # LEVEL B
            start_b = time.time()
            res_b = llm.invoke(question)
            end_b = time.time()
            score_b = calculate_score(res_b, ground_truth)

            # LEVEL C
            docs = db.similarity_search(question, k=3)
            context_text = "\n\n".join([d.page_content for d in docs])
            full_prompt = PROMPT_TEMPLATE.format(retrieved_chunks=context_text, user_question=question)

            start_c = time.time()
            res_c = llm.invoke(full_prompt)
            end_c = time.time()
            score_c = calculate_score(res_c, ground_truth)

            latency = end_c - start_c
            tokens = len(res_c.split()) * 1.3
            tps = tokens / latency if latency > 0 else 0

            results.append({
                "Model": model_name,
                "Soru_ID": i + 1,
                "Level_B": round(score_b, 3),
                "Level_C": round(score_c, 3),
                "Süre": round(latency, 2),
                "Hız": round(tps, 1),
                "Donanım": get_hw_usage()
            })

    # --- GÜVENLİ KAYIT BLOĞU ---
    df_res = pd.DataFrame(results)

    # 1. Önce ekrana bas (Garanti olsun)
    print("\nSONUÇLAR (KAYIT BAŞARISIZ OLURSA BURADAN KOPYALA):")
    print(df_res.to_markdown())

    # 2. Excel Dene, Olmazsa CSV Yap
    output_excel = "RAG_Final_Benchmark.xlsx"
    output_csv = "RAG_Final_Benchmark.csv"

    try:
        df_res.to_excel(output_excel, index=False)
        print(f"\nExcel kaydedildi: {output_excel}")
    except Exception as e:
        print(f"\nExcel hatası ({e}). CSV olarak kaydediliyor...")
        df_res.to_csv(output_csv, index=False, sep=';')
        print(f"CSV kaydedildi: {output_csv}")


if __name__ == "__main__":
    run_evaluation()