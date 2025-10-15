"""Prefect flow definitions for Freshbot."""

from .agents import (
    auditor_agent_flow,
    code_executor_agent_flow,
    custom_agent_flow,
    planner_agent_flow,
    responder_agent_flow,
    tool_user_agent_flow,
)
from .heartbeat import heartbeat_flow
from .ingest import (
    FLOW_REGISTRY as ingest_flow_registry,
    ingest_code_flow,
    ingest_general_flow,
    ingest_law_flow,
    ingest_pipeline_flow,
)
from .ingestion import freshbot_document_ingest
from .tasks import (
    parse_task_document_flow,
    sync_task_document_flow,
    sync_task_document_from_kb_flow,
    sync_task_graph_to_neo4j_flow,
)
from .tools import (
    agent_catalog_tool,
    agent_registry_snapshot_tool,
    code_embedding_tool,
    graph_capabilities_map_tool,
    gemini_chat_tool,
    kb_search_tool,
    prefect_flow_catalog_tool,
    qwen_chat_tool,
    search_agents_tool,
    search_tools_tool,
    tool_manifest_fetch_tool,
)
from .metadata import metadata_flag_flow

__all__ = [
    "agent_registry_snapshot_tool",
    "agent_catalog_tool",
    "auditor_agent_flow",
    "code_embedding_tool",
    "code_executor_agent_flow",
    "custom_agent_flow",
    "freshbot_document_ingest",
    "ingest_code_flow",
    "ingest_general_flow",
    "ingest_law_flow",
    "ingest_pipeline_flow",
    "ingest_flow_registry",
    "graph_capabilities_map_tool",
    "gemini_chat_tool",
    "heartbeat_flow",
    "kb_search_tool",
    "metadata_flag_flow",
    "prefect_flow_catalog_tool",
    "parse_task_document_flow",
    "planner_agent_flow",
    "qwen_chat_tool",
    "responder_agent_flow",
    "search_agents_tool",
    "search_tools_tool",
    "sync_task_graph_to_neo4j_flow",
    "sync_task_document_flow",
    "sync_task_document_from_kb_flow",
    "tool_manifest_fetch_tool",
    "tool_user_agent_flow",
]
