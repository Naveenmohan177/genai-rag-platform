import time
from typing import Dict, Any
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Initialize OpenTelemetry Tracer
provider = TracerProvider()
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("genai.rag.tracer")

class RAGSpanTracer:
    @staticmethod
    def trace_rag_pipeline(query: str, chunks_count: int, latency_ms: float, groundedness: float) -> Dict[str, Any]:
        """Emits OpenTelemetry GenAI semantic convention spans for RAG execution."""
        with tracer.start_as_current_span("genai.rag.workflow") as main_span:
            main_span.set_attribute("genai.operation.name", "rag_query")
            main_span.set_attribute("genai.user.query", query)
            main_span.set_attribute("genai.response.latency_ms", latency_ms)

            # Sub-span 1: Vector Search Stage
            with tracer.start_as_current_span("vector_db.faiss.search") as faiss_span:
                faiss_span.set_attribute("db.system", "faiss")
                faiss_span.set_attribute("db.retrieved_chunks", chunks_count)

            # Sub-span 2: Groundedness Evaluation Stage
            with tracer.start_as_current_span("eval.groundedness_check") as eval_span:
                eval_span.set_attribute("eval.metric.groundedness", groundedness)
                eval_span.set_attribute("eval.status", "passed" if groundedness > 0.7 else "failed")

            return {
                "trace_id": hex(main_span.get_span_context().trace_id),
                "span_id": hex(main_span.get_span_context().span_id),
                "status": "Telemetry Span Exported"
            }
