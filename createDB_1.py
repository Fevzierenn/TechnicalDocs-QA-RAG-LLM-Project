import os
import shutil
import time
import json
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # Updated library
from langchain_huggingface import HuggingFaceEmbeddings  # Updated library
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, Language

# 1. SETTINGS AND PATH CONFIGURATION

print("--- PHASE 1: DATABASE GENERATION ENGINE ---")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_PATH = os.path.join(script_dir, "baeldung_articles_markdown")

# Create folder if it doesn't exist (For testing)
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
    print(f"WARNING: '{DATA_PATH}' folder not found and created. Please place .md files here.")

# Embedding Models requested by the guide
EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5"
}

# Dictionary to hold results for reporting
benchmark_metrics = {}



# 2. DOCUMENT LOADING AND CLEANING

def load_documents():
    print(f"\n--> Step 1: Reading files: {DATA_PATH}")
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8', 'autodetect_encoding': True},
        show_progress=True
    )
    docs = loader.load()
    if not docs:
        print("ERROR: No files could be loaded! Check the folder path.")
        exit()

    print(f"{len(docs)} articles loaded into memory.")
    return docs



# 3. CHUNKING STRATEGIES


def get_chunks_strategy_fixed(docs):
    """STRATEGY A: FIXED-LENGTH (Guide: Fixed-length chunking)"""
    # 1000 tokens is approximately 4000 characters, overlap is important for context continuity
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)


def get_chunks_strategy_smart(docs):
    """STRATEGY B: SECTION-BASED (Guide: Section-based chunking)"""
    # Splitting by Markdown headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

    md_header_splits = []
    for doc in docs:
        # Metadata might be lost during Markdown split, adding it manually
        splits = markdown_splitter.split_text(doc.page_content)
        for split in splits:
            split.metadata['source'] = doc.metadata.get('source', 'unknown')
        md_header_splits.extend(splits)

    # If headers are too long, split them with fixed as well (Hybrid approach)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(md_header_splits)



# 4. VECTOR DATABASE GENERATOR

def get_chunks_strategy_optimized(docs):
    """
    STRATEGY B+: ENHANCED SECTION-BASED
    Optimized for Java Technical Articles (Baeldung style)
    """


    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]


    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    md_header_splits = []
    for doc in docs:
        splits = markdown_splitter.split_text(doc.page_content)
        for split in splits:
            split.metadata['source'] = doc.metadata.get('source', 'unknown')
        md_header_splits.extend(splits)
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA,
        chunk_size=1000,
        chunk_overlap=200
    )
    final_chunks = text_splitter.split_documents(md_header_splits)

    return final_chunks


def create_vector_db(docs, chunk_strategy, embed_model_key):
    db_name = f"vector_db_{chunk_strategy}_{embed_model_key}"
    db_folder_path = os.path.join(script_dir, "vector_dbs", db_name)

    # Chunking Process
    if chunk_strategy == "fixed":
        chunks = get_chunks_strategy_fixed(docs)
    else:
        #chunks = get_chunks_strategy_smart(docs)
        chunks = get_chunks_strategy_optimized(docs)

    print(f"\n------------------------------------------------")
    print(f"GENERATING: {db_name}")
    print(f"Model: {EMBEDDING_MODELS[embed_model_key]}")
    print(f"Chunk Count: {len(chunks)}")

    # Load Embedding Model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODELS[embed_model_key],
        model_kwargs={'device': 'cpu'},  # You can change to 'cuda' if GPU is available
        encode_kwargs={'normalize_embeddings': True}
    )

    # Guide request: Save embedding dimension
    test_embed = embeddings.embed_query("test")
    embed_dim = len(test_embed)

    # Clean old DB if exists
    if os.path.exists(db_folder_path):
        shutil.rmtree(db_folder_path)

    # START STOPWATCH (Guide: Encoding Time)
    start_time = time.time()

    # Create ChromaDB (Persistent)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_folder_path
    )

    end_time = time.time()
    duration = end_time - start_time

    print(f"SUCCESS.")
    print(f"ENCODING TIME: {duration:.2f} seconds")
    print(f"EMBEDDING DIM: {embed_dim}")

    # Save Results for Report
    benchmark_metrics[db_name] = {
        "chunk_strategy": chunk_strategy,
        "embedding_model": embed_model_key,
        "encoding_time_sec": round(duration, 2),
        "embedding_dimension": embed_dim,
        "total_chunks": len(chunks)
    }



# 5. MAIN

if __name__ == "__main__":
    # Prepare folder structure
    if not os.path.exists(os.path.join(script_dir, "vector_dbs")):
        os.makedirs(os.path.join(script_dir, "vector_dbs"))

    raw_docs = load_documents()

    if raw_docs:
        # SCENARIO 1: Smart Chunking + MiniLM (Fast, Small)
        create_vector_db(raw_docs, chunk_strategy="smart", embed_model_key="minilm")

        # SCENARIO 2: Fixed Chunking + MiniLM (Control Group 1)
        create_vector_db(raw_docs, chunk_strategy="fixed", embed_model_key="minilm")

        # SCENARIO 3: Smart Chunking + BGE (Claims better retrieval)
        create_vector_db(raw_docs, chunk_strategy="smart", embed_model_key="bge")

        # Save metrics as JSON (Golden value for the report table)
        with open(os.path.join(script_dir, "benchmark_metrics.json"), "w") as f:
            json.dump(benchmark_metrics, f, indent=4)

        print("\nALL DATABASES CREATED AND METRICS SAVED!")
        print(f"File location: {os.path.join(script_dir, 'vector_dbs')}")