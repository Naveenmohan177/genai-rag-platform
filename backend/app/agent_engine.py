import os
import io
from typing import Dict, Any, List
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from app.guardrails import SecurityGuardrails
from openai import OpenAI as DirectOpenAI

class MultimodalAgentEngine:
    def __init__(self, openai_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.client = DirectOpenAI(api_key=openai_api_key)
        self.faiss_db = None
        self.bm25_retriever = None

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribes incoming audio streams using OpenAI Whisper API."""
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "input_audio.wav"
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript.text

    def index_documents(self, file_paths: List[str]) -> int:
        documents = []
        for path in file_paths:
            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        chunks = text_splitter.split_documents(documents)

        self.faiss_db = FAISS.from_documents(chunks, self.embeddings)
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = 4

        return len(chunks)

    def _reciprocal_rank_fusion(self, dense_docs: List, sparse_docs: List, k: int = 60) -> List:
        scores = {}
        doc_map = {}

        for rank, doc in enumerate(dense_docs):
            content = doc.page_content
            scores[content] = scores.get(content, 0) + (1 / (k + rank + 1))
            doc_map[content] = doc

        for rank, doc in enumerate(sparse_docs):
            content = doc.page_content
            scores[content] = scores.get(content, 0) + (1 / (k + rank + 1))
            doc_map[content] = doc

        sorted_contents = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[c] for c in sorted_contents[:3]]

    def process_agent_query(self, user_query: str) -> Dict[str, Any]:
        if SecurityGuardrails.check_prompt_injection(user_query):
            return {
                "answer": "Security Policy Violation: Prompt injection blocked.",
                "sources": [],
                "confidence": "Blocked",
                "passed_validation": False
            }

        sanitized_query = SecurityGuardrails.sanitize_input(user_query)

        sources = []
        context_str = ""
        if self.faiss_db and self.bm25_retriever:
            dense_results = [doc for doc, _ in self.faiss_db.similarity_search_with_score(sanitized_query, k=4)]
            sparse_results = self.bm25_retriever.invoke(sanitized_query)
            reranked_docs = self._reciprocal_rank_fusion(dense_results, sparse_results)
            context_str = "\n\n".join([d.page_content for d in reranked_docs])
            sources = [d.page_content for d in reranked_docs]

        prompt_template = """
        You are an Autonomous Systems & Knowledge Agent.
        Provide a concise, direct response suitable for spoken output.

        Retrieved Knowledge Context:
        {context}

        User Request: {question}
        Response:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        response = self.llm.invoke(prompt.format(context=context_str if context_str else "No Index Context", question=sanitized_query)).content

        return {
            "answer": response,
            "sources": sources,
            "confidence": "High (Autonomous Agent)",
            "passed_validation": True
        }
