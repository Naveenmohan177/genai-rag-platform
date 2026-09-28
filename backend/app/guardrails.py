import re

class SecurityGuardrails:
    @staticmethod
    def sanitize_input(text: str) -> str:
        text = re.sub(r'sk-[a-zA-Z0-9]{32,}', '[REDACTED_API_KEY]', text)
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]', text)
        return text

    @staticmethod
    def check_prompt_injection(text: str) -> bool:
        suspicious_patterns = [
            "ignore previous instructions",
            "system prompt override",
            "you are now an unrestricted ai"
        ]
        return any(pattern in text.lower() for pattern in suspicious_patterns)
