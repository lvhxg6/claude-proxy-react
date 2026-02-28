"""Compaction support for Claude Code CLI context management.

When conversations approach the context window limit, Claude Code CLI sends
a context_management.edits field requesting compaction. This module handles
detecting the trigger, building summarization requests, and constructing
the compacted follow-up request.
"""

import json
import uuid
import logging
from typing import Optional, Dict, Any, List, Tuple

from src.core.config import config
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest, CompactionEdit

logger = logging.getLogger(__name__)

DEFAULT_COMPACTION_PROMPT = (
    "You are a conversation summarizer. Summarize the following conversation "
    "concisely while preserving all important context, decisions, code changes, "
    "file paths, and technical details. The summary will be used as context for "
    "continuing the conversation. Be thorough but concise.\n\n"
    "CONVERSATION:\n"
)


def should_compact(
    request: ClaudeMessagesRequest, effective_input: int
) -> Optional[CompactionEdit]:
    """Check if compaction should be triggered.

    Strategy:
    1) 尊重客户端的 compaction edit，但把触发值上限限定在模型窗口比例内。
    2) 若客户端未提供或未触发，在逼近窗口时自动触发，避免 400。
    """
    if not config.compaction_enabled:
        return None

    auto_trigger = int(config.model_context_window * config.compaction_trigger_ratio)

    # 1) 客户端带了 context_management 的情况
    if request.context_management:
        for edit in request.context_management.edits:
            if edit.type == "compact_20260112":
                client_trigger = edit.trigger.value if edit.trigger else 150000
                effective_trigger = min(client_trigger, auto_trigger)
                if effective_input >= effective_trigger:
                    logger.info(
                        "Compaction triggered (client edit): "
                        f"effective_input={effective_input} >= trigger={effective_trigger} "
                        f"(client={client_trigger}, auto_cap={auto_trigger})"
                    )
                    return edit

    # 2) 无 context_management 或未触发时的自动压缩
    #    使用 pause_after_compaction=False，让 proxy 自己完成压缩+继续请求，
    #    不依赖 CLI 重建上下文（CLI 可能不知道如何处理 auto compaction 的 pause 响应）
    if effective_input >= auto_trigger:
        logger.info(
            "Compaction triggered (auto): "
            f"effective_input={effective_input} >= auto_trigger={auto_trigger}"
        )
        return CompactionEdit(pause_after_compaction=False)

    return None


def _serialize_messages(request: ClaudeMessagesRequest) -> str:
    """Serialize conversation messages to text for summarization."""
    parts = []

    # Include system prompt context
    if request.system:
        if isinstance(request.system, str):
            parts.append(f"[System]: {request.system}")
        elif isinstance(request.system, list):
            for block in request.system:
                if hasattr(block, "text"):
                    parts.append(f"[System]: {block.text}")

    for msg in request.messages:
        role = msg.role.upper()
        if isinstance(msg.content, str):
            parts.append(f"[{role}]: {msg.content}")
        elif isinstance(msg.content, list):
            for block in msg.content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        parts.append(f"[{role}]: {block.text}")
                    elif block.type == "tool_use":
                        parts.append(
                            f"[{role} tool_use]: {block.name}({json.dumps(block.input, ensure_ascii=False)[:500]})"
                        )
                    elif block.type == "tool_result":
                        content = block.content if isinstance(block.content, str) else json.dumps(block.content, ensure_ascii=False)[:500]
                        parts.append(f"[{role} tool_result]: {content}")

    return "\n".join(parts)


def build_compaction_messages(
    request: ClaudeMessagesRequest, compaction_edit: CompactionEdit
) -> List[Dict[str, Any]]:
    """Build OpenAI-format messages for the summarization request."""
    conversation_text = _serialize_messages(request)

    # Truncate if too long (leave room for the prompt itself)
    max_chars = config.model_context_window * 3  # rough char-to-token ratio
    if len(conversation_text) > max_chars:
        conversation_text = conversation_text[-max_chars:]
        logger.warning(
            f"Compaction: truncated conversation to last {max_chars} chars"
        )

    if compaction_edit.instructions:
        prompt = compaction_edit.instructions + "\n\nCONVERSATION:\n" + conversation_text
    else:
        prompt = DEFAULT_COMPACTION_PROMPT + conversation_text

    return [
        {"role": "system", "content": "You are a helpful assistant that summarizes conversations accurately and concisely."},
        {"role": "user", "content": prompt},
    ]


def build_followup_request(
    summary: str,
    request: ClaudeMessagesRequest,
    mapped_model: str,
    openai_request: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a new OpenAI request using the compaction summary as context.

    Keeps system prompt, replaces messages with [assistant: summary, user: last message].
    """
    # Extract system messages from original openai_request
    system_messages = [
        m for m in openai_request.get("messages", [])
        if m.get("role") == Constants.ROLE_SYSTEM
    ]

    # Get the last user message from original request
    last_user_content = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            if isinstance(msg.content, str):
                last_user_content = msg.content
            elif isinstance(msg.content, list):
                text_parts = []
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "text":
                        text_parts.append(block.text)
                last_user_content = "\n".join(text_parts)
            break

    # Build new messages
    new_messages = list(system_messages)
    new_messages.append({"role": "assistant", "content": summary})
    new_messages.append({"role": "user", "content": last_user_content})

    # Build new request, copying relevant fields
    new_request = {
        "model": mapped_model,
        "messages": new_messages,
        "max_tokens": openai_request.get("max_tokens", config.max_tokens_limit),
        "temperature": openai_request.get("temperature", 1.0),
        "stream": openai_request.get("stream", False),
    }

    # Copy optional fields
    if "tools" in openai_request:
        new_request["tools"] = openai_request["tools"]
    if "tool_choice" in openai_request:
        new_request["tool_choice"] = openai_request["tool_choice"]
    if "stop" in openai_request:
        new_request["stop"] = openai_request["stop"]
    if "top_p" in openai_request:
        new_request["top_p"] = openai_request["top_p"]

    return new_request


def build_compaction_response(
    summary: str,
    request: ClaudeMessagesRequest,
    compaction_input_tokens: int,
    compaction_output_tokens: int,
) -> Dict[str, Any]:
    """Build a Claude-format response containing only the compaction block.

    Used when pause_after_compaction=true.
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    return {
        "id": message_id,
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": request.model,
        "content": [
            {"type": Constants.CONTENT_COMPACTION, "content": summary}
        ],
        "stop_reason": Constants.STOP_COMPACTION,
        "stop_sequence": None,
        "usage": {
            "input_tokens": compaction_input_tokens,
            "output_tokens": compaction_output_tokens,
            "iterations": [
                {
                    "type": "compaction",
                    "input_tokens": compaction_input_tokens,
                    "output_tokens": compaction_output_tokens,
                }
            ],
        },
    }


def build_compaction_with_response(
    summary: str,
    content_blocks: list,
    request: ClaudeMessagesRequest,
    stop_reason: str,
    compaction_input_tokens: int,
    compaction_output_tokens: int,
    message_input_tokens: int,
    message_output_tokens: int,
) -> Dict[str, Any]:
    """Build a Claude-format response with compaction block + normal content.

    Used when pause_after_compaction=false.
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    all_content = [
        {"type": Constants.CONTENT_COMPACTION, "content": summary}
    ] + content_blocks

    return {
        "id": message_id,
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": request.model,
        "content": all_content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": message_input_tokens,
            "output_tokens": compaction_output_tokens + message_output_tokens,
            "iterations": [
                {
                    "type": "compaction",
                    "input_tokens": compaction_input_tokens,
                    "output_tokens": compaction_output_tokens,
                },
                {
                    "type": "message",
                    "input_tokens": message_input_tokens,
                    "output_tokens": message_output_tokens,
                },
            ],
        },
    }
