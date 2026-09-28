import os
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from app.guardrails import SecurityGuardrails

class EnterpriseRAGEngine:
    def __init__(self, openai_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.faiss_db = None
        self.bm25_retriever = None

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
        self.bm25_retriever.k = 5

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
        return [doc_map[c] for c in sorted_contents[:4]]

    def query(self, user_query: str, similarity_threshold: float = 0.65) -> Dict[str, Any]:
        if not self.faiss_db or not self.bm25_retriever:
            return {"answer": "No documents indexed.", "sources": [], "confidence": "Low", "passed_validation": False}

        if SecurityGuardrails.check_prompt_injection(user_query):
            return {"answer": "Security Policy Violation: Prompt Injection Detected.", "sources": [], "confidence": "Blocked", "passed_validation": False}

        sanitized_query = SecurityGuardrails.sanitize_input(user_query)

        dense_results = [doc for doc, _ in self.faiss_db.similarity_search_with_score(sanitized_query, k=5)]
        sparse_results = self.bm25_retriever.invoke(sanitized_query)
        reranked_docs = self._reciprocal_rank_fusion(dense_results, sparse_results)

        if not reranked_docs:
            return {"answer": "Context below confidence threshold.", "sources": [], "confidence": "Filtered", "passed_validation": False}

        context_str = "\n\n".join([d.page_content for d in reranked_docs])

        prompt_template = """
        You are an Enterprise System Architect.
        Answer strictly using the provided context.
        If context is insufficient, reply: "Insufficient Context Found."

        Context:
        {context}

        Question: {question}
        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        response = self.llm.invoke(prompt.format(context=context_str, question=sanitized_query)).content

        passed_validation = "Insufficient Context Found" not in response

        return {
            "answer": response,
            "sources": [d.page_content for d in reranked_docs],
            "confidence": "High (RRF Reranked)" if passed_validation else "Filtered",
            "passed_validation": passed_validation
        }
