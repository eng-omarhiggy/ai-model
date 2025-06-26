import os
import PyPDF2
import re
import json
import csv

# Folder where your source files are stored
SOURCE_DIR = "./data_sources"  # Change this path if needed
VAULT_FILE = "vault.txt"

# Chunking helper
def chunk_text(text, max_chunk_size=1000):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < max_chunk_size:
            current_chunk += (sentence + " ").strip()
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# File writing (ensures one line per chunk)
def write_chunks(chunks):
    with open(VAULT_FILE, "a", encoding="utf-8") as f:
        for chunk in chunks:
            cleaned = chunk.replace('\n', ' ').replace('\r', ' ').strip()
            f.write(cleaned + "\n")
    print(f"✅ Appended {len(chunks)} chunks to {VAULT_FILE}")

# PDF processing
def process_pdf(file_path):
    print(f"Processing PDF: {file_path}")
    text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    text = re.sub(r'\s+', ' ', text).strip()
    write_chunks(chunk_text(text))

# TXT processing
def process_txt(file_path):
    print(f"Processing TXT: {file_path}")
    with open(file_path, 'r', encoding="utf-8") as f:
        text = re.sub(r'\s+', ' ', f.read()).strip()
    write_chunks(chunk_text(text))

# JSON processing
def process_json(file_path):
    print(f"Processing JSON: {file_path}")
    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        text = json.dumps(data, ensure_ascii=False)
        text = re.sub(r'\s+', ' ', text).strip()
    write_chunks(chunk_text(text))

# CSV processing with text + link columns
def process_csv(file_path):
    print(f"Processing CSV: {file_path}")
    chunks = []
    with open(file_path, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        for row_num, row in enumerate(reader):
            if len(row) < 2:
                print(f"Skipping malformed row {row_num}: {row}")
                continue
            text = row[0].strip()
            link = row[1].strip()
            if text and link:
                full_text = f"{text} [Source: {link}]"
                chunks += chunk_text(full_text)
    write_chunks(chunks)

# Dispatcher for folder files
def process_directory(directory):
    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        if file.lower().endswith('.pdf'):
            process_pdf(full_path)
        elif file.lower().endswith('.txt'):
            process_txt(full_path)
        elif file.lower().endswith('.json'):
            process_json(full_path)
        elif file.lower().endswith('.csv'):
            process_csv(full_path)

# Entry point
if __name__ == "__main__":
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Source directory '{SOURCE_DIR}' does not exist.")
    else:
        process_directory(SOURCE_DIR)
