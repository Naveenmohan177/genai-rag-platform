import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class MultiAgentWorkflow:
    def __init__(self, openai_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    def architect_agent(self, query: str, context: str) -> str:
        prompt = PromptTemplate(
            template="You are a Senior System Architect. Draft a comprehensive architecture response.\nContext:\n{context}\n\nQuery: {query}\nDraft Response:",
            input_variables=["context", "query"]
        )
        return self.llm.invoke(prompt.format(context=context, query=query)).content

    def security_compliance_agent(self, draft_response: str) -> Dict[str, Any]:
        prompt = PromptTemplate(
            template="You are a DevSecOps Security Auditor. Review this response for compliance, data leaks, or vulnerabilities.\nResponse Draft:\n{draft}\n\nProvide AUDIT_STATUS (PASSED/FLAGGED) and AUDIT_NOTES:",
            input_variables=["draft"]
        )
        audit_output = self.llm.invoke(prompt.format(draft=draft_response)).content
        passed = "FLAGGED" not in audit_output.upper()
        return {"passed": passed, "audit_report": audit_output}

    def execute_workflow(self, query: str, context: str) -> Dict[str, Any]:
        # Step 1: Architect Agent generates response
        draft = self.architect_agent(query, context)
        
        # Step 2: Security Agent performs compliance audit
        audit = self.security_compliance_agent(draft)

        return {
            "draft_response": draft,
            "security_passed": audit["passed"],
            "security_notes": audit["audit_report"],
            "status": "AWAITING_HUMAN_APPROVAL" if audit["passed"] else "BLOCKED_BY_SECURITY"
        }
