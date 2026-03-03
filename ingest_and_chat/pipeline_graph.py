# ingest_and_chat/pipeline_graph.py
"""
Builds the multi-agent RAG ingestion StateGraph.

Pipeline (sequential with error short-circuit at each step):
    read_and_classify_files
    process_text_documents
    process_media_files
    process_structured_files
    generate_metadata
    setup_postgres
    vectorize_and_store
"""

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from .states import RAGIngestionState
from . import nodes


def _check_error(state: RAGIngestionState) -> str:
    """Routing function: if has_error is True, stop the graph."""
    if state.get("has_error", False):
        return "stop"
    return "continue"


def create_ingestion_graph():
    """Builds and compiles the ingestion graph with in-memory checkpointing."""
    memory = InMemorySaver()
    builder = StateGraph(RAGIngestionState)

    # -- Add Nodes --
    builder.add_node("read_and_classify_files", nodes.read_and_classify_files)
    builder.add_node("process_text_documents", nodes.process_text_documents)
    builder.add_node("process_media_files", nodes.process_media_files)
    builder.add_node("process_structured_files", nodes.process_structured_files)
    builder.add_node("generate_metadata", nodes.generate_metadata)
    builder.add_node("setup_postgres", nodes.setup_postgres)
    builder.add_node("vectorize_and_store", nodes.vectorize_and_store)

    # -- Add Edges --
    builder.add_edge(START, "read_and_classify_files")

    builder.add_conditional_edges(
        "read_and_classify_files", _check_error,
        {"stop": END, "continue": "process_text_documents"},
    )
    builder.add_conditional_edges(
        "process_text_documents", _check_error,
        {"stop": END, "continue": "process_media_files"},
    )
    builder.add_conditional_edges(
        "process_media_files", _check_error,
        {"stop": END, "continue": "process_structured_files"},
    )
    builder.add_conditional_edges(
        "process_structured_files", _check_error,
        {"stop": END, "continue": "generate_metadata"},
    )
    builder.add_conditional_edges(
        "generate_metadata", _check_error,
        {"stop": END, "continue": "setup_postgres"},
    )
    builder.add_conditional_edges(
        "setup_postgres", _check_error,
        {"stop": END, "continue": "vectorize_and_store"},
    )

    builder.add_edge("vectorize_and_store", END)

    return builder.compile(checkpointer=memory)
