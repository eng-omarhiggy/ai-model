import torch
import ollama
import os
from openai import OpenAI
import argparse
import json

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Get relevant chunks based on similarity
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if vault_embeddings.nelement() == 0:
        return []
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

# Optional: rephrase user query using past conversation
def rewrite_query(user_input_json, conversation_history, ollama_model):
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
The rewritten query should:
- Preserve the core intent and meaning
- Clarify it for document search
- DO NOT answer the query
- Output ONLY the rewritten query

Conversation History:
{context}

Original query: [{user_input}]

Rewritten query:
"""
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )
    rewritten_query = response.choices[0].message.content.strip()
    return json.dumps({"Rewritten Query": rewritten_query})

# Main chat logic
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})

    if len(conversation_history) > 1:
        query_json = {"Query": user_input, "Rewritten Query": ""}
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        print(PINK + "Original Query: " + user_input + RESET_COLOR)
        print(PINK + "Rewritten Query: " + rewritten_query + RESET_COLOR)
    else:
        rewritten_query = user_input

    relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)
        response_text = "I'm sorry, I don't have enough information in the provided documents to answer that."
        conversation_history.append({"role": "assistant", "content": response_text})
        return response_text

    user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
    conversation_history[-1]["content"] = user_input_with_context

    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]

    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )

    answer = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": answer})
    return answer

# --- CLI args
print(NEON_GREEN + "Parsing command-line arguments..." + RESET_COLOR)
parser = argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default="llama3", help="Ollama model to use (default: llama3)")
args = parser.parse_args()

# --- API setup
print(NEON_GREEN + "Initializing Ollama API client..." + RESET_COLOR)
client = OpenAI(base_url='http://localhost:11434/v1', api_key='llama3')

# --- Load vault
print(NEON_GREEN + "Loading vault content..." + RESET_COLOR)
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        for line in vault_file:
            cleaned = line.replace('\n', ' ').replace('\r', ' ').strip()
            vault_content.append(cleaned)

# --- Embed vault
print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
vault_embeddings = []
for content in vault_content:
    response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
    vault_embeddings.append(response["embedding"])
vault_embeddings_tensor = torch.tensor(vault_embeddings)
print("✅ Embeddings loaded.")

# --- System prompt (strict mode)
system_message = (
    "You are a document-bound assistant. You must only answer using the provided context. "
    "If the answer is not in the documents, respond with:\n"
    "'I'm sorry, I don't have enough information in the provided documents to answer that.'\n"
    "Never use your own knowledge or assumptions. Refer to links in context if present."
)

# --- Chat loop
print("🟢 Starting conversation loop...")
conversation_history = []

while True:
    user_input = input(YELLOW + "Ask a query about your documents (or type 'quit' to exit): " + RESET_COLOR)
    if user_input.lower() == 'quit':
        break

    response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
    print(NEON_GREEN + "Response:\n\n" + response + RESET_COLOR)
