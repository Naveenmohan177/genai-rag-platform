import os
import json
import re
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class AdvancedDataPipeline:
    def __init__(self, output_dir: str = "./data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def clean_and_sanitize_text(self, text: str) -> str:
        """Cleans raw document text and removes sensitive PII before ingestion or training."""
        text = re.sub(r'<[^>]+>', '', text)  # Strip HTML tags
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)  # Redact emails
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]', text)  # Redact phones
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text

    def process_raw_documents(self, file_paths: List[str]) -> str:
        """Loads, cleans, and outputs standardized dataset JSON for vector search or training."""
        cleaned_docs = []
        for path in file_paths:
            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path, encoding="utf-8")
            
            raw_docs = loader.load()
            for doc in raw_docs:
                cleaned_content = self.clean_and_sanitize_text(doc.page_content)
                cleaned_docs.append({"source": path, "content": cleaned_content})

        output_file = os.path.join(self.output_dir, "processed_knowledge.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(cleaned_docs, f, indent=2)
        
        return output_file

    def generate_openai_finetuning_jsonl(self, json_file: str, output_jsonl: str = "finetune_dataset.jsonl") -> str:
        """Converts cleaned docs into OpenAI fine-tuning JSONL format (Prompt-Response pairs)."""
        with open(json_file, "r", encoding="utf-8") as f:
            docs = json.load(f)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        jsonl_path = os.path.join(self.output_dir, output_jsonl)

        dataset = []
        for item in docs:
            chunks = text_splitter.split_text(item["content"])
            for idx, chunk in enumerate(chunks[:5]):  # Process sample chunks
                fine_tune_entry = {
                    "messages": [
                        {"role": "system", "content": "You are an Enterprise System Architect assistant."},
                        {"role": "user", "content": f"Summarize key technical requirements from context chunk {idx+1}."},
                        {"role": "assistant", "content": chunk}
                    ]
                }
                dataset.append(fine_tune_entry)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for entry in dataset:
                f.write(json.dumps(entry) + "\n")

        return jsonl_path

if __name__ == "__main__":
    pipeline = AdvancedDataPipeline()
    print("Data Engineering & Fine-Tuning Pipeline initialized successfully.")
