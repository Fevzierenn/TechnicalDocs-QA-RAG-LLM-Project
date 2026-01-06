import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os

# --- SETTINGS ---
BASE_PATH = r"D:\LLM_PROJECT-DEEPLEARNING\vector_dbs"
CSV_PATH = r"D:\LLM_PROJECT-DEEPLEARNING\golden-QA.csv"
COMMON_COLLECTION_NAME = "langchain"

# Models to Compare
DBS_TO_COMPARE = {
    "1. Old (Fixed Strategy)": "vector_db_fixed_minilm",
    "2. New (Smart/Optimized)": "vector_db_smart_minilm"
}

# Embedding model
minilm_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def inspect_retrieval(question_text, answer_text, n_results=3):
    print(f"\n{'=' * 100}")
    print(f"Question: {question_text}")
    print(f"Truth Answer (Summary): {str(answer_text)[:120]}...")
    print(f"{'=' * 100}")

    for display_name, folder_name in DBS_TO_COMPARE.items():
        full_db_path = os.path.join(BASE_PATH, folder_name)
        print(f"\n   >>> MODEL: {display_name}")

        if not os.path.exists(full_db_path):
            print(f"      ALERT: File not found -> {full_db_path}")
            continue

        try:
            client = chromadb.PersistentClient(path=full_db_path)
            collection = client.get_collection(name=COMMON_COLLECTION_NAME, embedding_function=minilm_ef)

            results = collection.query(
                query_texts=[question_text],
                n_results=n_results
            )

            # List results
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                dist = results['distances'][0][i]  # L2 Distance: much better when close to 0 zero.

                source_file = os.path.basename(meta.get('source', 'Unknown'))
                h1 = meta.get('Header 1', '')
                h2 = meta.get('Header 2', '')
                section_info = f"{h1} > {h2}" if h1 or h2 else "No Section Info"

                print(f"   [Rank {i + 1} | Distance Score: {dist:.4f}]")
                print(f" Source: {source_file}")
                print(f" Section:  {section_info}")
                print(f" Content: \"{doc.replace(chr(10), ' ')[:250]}...\"")
                print("   " + "-" * 40)

        except Exception as e:
            print(f"      Error: {e}")


# --- MAIN ---
if __name__ == "__main__":
    print("--- Starting Top 3 Chunk Analysis...3 ---")

    try:
        try:
            df = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(CSV_PATH, sep=';', encoding='cp1252')

        df.columns = df.columns.str.strip().str.lower()

        # Kept 'soru' and 'cevap' checks in case the CSV still has Turkish headers
        q_col = next((c for c in df.columns if 'question' in c or 'soru' in c), None)
        a_col = next((c for c in df.columns if 'answer' in c or 'cevap' in c), None)

        if not q_col: raise ValueError("Question column not found!")

    except Exception as e:
        print(f"CSV Error: {e}")
        exit()

    # Random Questions
    sample_df = df.sample(3)

    for index, row in sample_df.iterrows():
        inspect_retrieval(row[q_col], row[a_col] if a_col else "N/A")
        input("\nPress ENTER to proceed to the next question...")

    print("\nAnalysis complete.")