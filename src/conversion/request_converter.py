import json
from typing import Dict, Any, List
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest, ClaudeMessage, ClaudeContentBlockCompaction
from src.core.config import config
from src.core.tokenizer import GLMTokenizer
import logging

logger = logging.getLogger(__name__)


def count_message_tokens(messages: List[Dict[str, Any]], model: str) -> int:
    """
    Count tokens in OpenAI format messages using GLM-4.7 tokenizer.

    Args:
        messages: List of OpenAI format messages
        model: Model name (unused, kept for compatibility)

    Returns:
        Exact token count for GLM-4.7
    """
    try:
        tokenizer = GLMTokenizer.get_tokenizer()
        num_tokens = 0

        for message in messages:
            # Message structure overhead (role markers, formatting)
            num_tokens += 4  # Base tokens per message

            for key, value in message.items():
                if key == "role":
                    num_tokens += len(tokenizer.encode(value).ids)
                elif key == "content":
                    if isinstance(value, str):
                        num_tokens += len(tokenizer.encode(value).ids)
                    elif isinstance(value, list):
                        # Handle multimodal content
                        for item in value:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    text = item.get("text", "")
                                    num_tokens += len(tokenizer.encode(text).ids)
                                elif item.get("type") == "image_url":
                                    # Image token estimation (GLM-4V uses ~85 tokens per image)
                                    num_tokens += 85
                elif key == "name":
                    num_tokens += len(tokenizer.encode(value).ids)
                elif key == "tool_calls":
                    # Count tokens in tool calls
                    for tool_call in value:
                        if isinstance(tool_call, dict):
                            tool_json = json.dumps(tool_call, ensure_ascii=False)
                            num_tokens += len(tokenizer.encode(tool_json).ids)
                elif key == "tool_call_id":
                    num_tokens += len(tokenizer.encode(value).ids)

        num_tokens += 2  # Reply priming tokens

        return num_tokens

    except Exception as e:
        logger.error(f"Error counting tokens with GLM tokenizer: {e}")
        # Fallback: character-based estimation
        total_chars = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total_chars += len(item.get("text", ""))
        return max(1, total_chars // 4)


def count_tools_tokens(tools: List[Dict[str, Any]], model: str) -> int:
    """Count tokens consumed by tools/function definitions.

    Tools schema (name, description, parameters JSON schema) are sent as part
    of the prompt and consume input tokens, but were previously not counted.
    """
    try:
        tokenizer = GLMTokenizer.get_tokenizer()
        num_tokens = 0

        for tool in tools:
            # Each tool has structural overhead
            num_tokens += 4
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            params = func.get("parameters", {})

            if name:
                num_tokens += len(tokenizer.encode(name).ids)
            if desc:
                num_tokens += len(tokenizer.encode(desc).ids)
            if params:
                params_json = json.dumps(params, ensure_ascii=False)
                num_tokens += len(tokenizer.encode(params_json).ids)

        logger.debug(f"Tools token count: {num_tokens} ({len(tools)} tools)")
        return num_tokens

    except Exception as e:
        logger.error(f"Error counting tools tokens: {e}")
        # Fallback: estimate from JSON character count
        tools_json = json.dumps(tools, ensure_ascii=False)
        return max(1, len(tools_json) // 3)


def _find_compaction_boundary(messages: List[ClaudeMessage]) -> tuple:
    """Find the latest compaction block and return (boundary_index, summary).

    Scans messages in reverse to find the most recent assistant message
    containing a compaction block. Per Anthropic protocol, all messages
    before this boundary should be discarded.

    Returns:
        (boundary_index, summary_text) if found, (-1, None) otherwise.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.role != Constants.ROLE_ASSISTANT:
            continue
        if isinstance(msg.content, str):
            continue
        if not isinstance(msg.content, list):
            continue
        for block in msg.content:
            if isinstance(block, ClaudeContentBlockCompaction) or (
                hasattr(block, "type") and block.type == "compaction"
            ):
                summary = getattr(block, "content", None)
                return i, summary
    return -1, None


def _prune_messages_at_boundary(
    messages: List[ClaudeMessage], boundary_index: int, summary: str | None
) -> List[ClaudeMessage]:
    """Prune messages at compaction boundary.

    Keeps only messages from boundary_index onward. The compaction block
    itself is preserved in the assistant message (will be converted to text).
    """
    pruned = messages[boundary_index:]
    original_count = len(messages)
    pruned_count = original_count - len(pruned)
    logger.info(
        f"Compaction boundary prune: dropped {pruned_count}/{original_count} messages, "
        f"kept {len(pruned)} from index {boundary_index}, "
        f"summary={'present' if summary else 'null (no-op)'}"
    )
    return pruned


def convert_claude_to_openai(
    claude_request: ClaudeMessagesRequest, model_manager
) -> Dict[str, Any]:
    """Convert Claude API request format to OpenAI format."""

    # Map model
    openai_model = model_manager.map_claude_model_to_openai(claude_request.model)

    # === Compaction boundary pruning ===
    # Per Anthropic protocol: find the latest compaction block and discard
    # all messages before it. The compaction summary replaces prior history.
    messages = list(claude_request.messages)
    boundary_index, summary = _find_compaction_boundary(messages)
    if boundary_index >= 0:
        messages = _prune_messages_at_boundary(messages, boundary_index, summary)

    # Convert messages
    openai_messages = []

    # Add system message if present
    if claude_request.system:
        system_text = ""
        if isinstance(claude_request.system, str):
            system_text = claude_request.system
        elif isinstance(claude_request.system, list):
            text_parts = []
            for block in claude_request.system:
                if hasattr(block, "type") and block.type == Constants.CONTENT_TEXT:
                    text_parts.append(block.text)
                elif (
                    isinstance(block, dict)
                    and block.get("type") == Constants.CONTENT_TEXT
                ):
                    text_parts.append(block.get("text", ""))
            system_text = "\n\n".join(text_parts)

        if system_text.strip():
            openai_messages.append(
                {"role": Constants.ROLE_SYSTEM, "content": system_text.strip()}
            )

    # Process Claude messages (using pruned messages if compaction boundary was found)
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.role == Constants.ROLE_USER:
            openai_message = convert_claude_user_message(msg)
            openai_messages.append(openai_message)
        elif msg.role == Constants.ROLE_ASSISTANT:
            openai_message = convert_claude_assistant_message(msg)
            openai_messages.append(openai_message)

            # Check if next message contains tool results
            if i + 1 < len(messages):
                next_msg = messages[i + 1]
                if (
                    next_msg.role == Constants.ROLE_USER
                    and isinstance(next_msg.content, list)
                    and any(
                        block.type == Constants.CONTENT_TOOL_RESULT
                        for block in next_msg.content
                        if hasattr(block, "type")
                    )
                ):
                    # Process tool results
                    i += 1  # Skip to tool result message
                    tool_results = convert_claude_tool_results(next_msg)
                    openai_messages.extend(tool_results)

        i += 1

    # Build OpenAI request
    openai_request = {
        "model": openai_model,
        "messages": openai_messages,
        "max_tokens": min(
            max(claude_request.max_tokens, config.min_tokens_limit),
            config.max_tokens_limit,
        ),
        "temperature": claude_request.temperature,
        "stream": claude_request.stream,
    }
    logger.debug(
        f"Converted Claude request to OpenAI format: {json.dumps(openai_request, indent=2, ensure_ascii=False)}"
    )
    # Add optional parameters
    if claude_request.stop_sequences:
        openai_request["stop"] = claude_request.stop_sequences
    if claude_request.top_p is not None:
        openai_request["top_p"] = claude_request.top_p

    # Convert tools
    if claude_request.tools:
        openai_tools = []
        for tool in claude_request.tools:
            if tool.name and tool.name.strip():
                openai_tools.append(
                    {
                        "type": Constants.TOOL_FUNCTION,
                        Constants.TOOL_FUNCTION: {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,
                        },
                    }
                )
        if openai_tools:
            openai_request["tools"] = openai_tools

    # Convert tool choice
    if claude_request.tool_choice:
        choice_type = claude_request.tool_choice.get("type")
        if choice_type == "auto":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "any":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "tool" and "name" in claude_request.tool_choice:
            openai_request["tool_choice"] = {
                "type": Constants.TOOL_FUNCTION,
                Constants.TOOL_FUNCTION: {"name": claude_request.tool_choice["name"]},
            }
        else:
            openai_request["tool_choice"] = "auto"

    # Calculate input tokens for context window checking
    # Count messages + tools schema (tools can consume significant tokens)
    input_tokens = count_message_tokens(openai_messages, openai_model)
    if "tools" in openai_request and openai_request["tools"]:
        input_tokens += count_tools_tokens(openai_request["tools"], openai_model)
    openai_request["_input_tokens"] = input_tokens

    # Store pruned messages for compaction summarization (if boundary was found)
    # This ensures compaction summaries are based on post-pruning context, not original history
    openai_request["_pruned_messages"] = messages

    return openai_request


def convert_claude_user_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude user message to OpenAI format."""
    if msg.content is None:
        return {"role": Constants.ROLE_USER, "content": ""}
    
    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_USER, "content": msg.content}

    # Handle multimodal content
    openai_content = []
    for block in msg.content:
        if block.type == Constants.CONTENT_TEXT:
            openai_content.append({"type": "text", "text": block.text})
        elif block.type == Constants.CONTENT_IMAGE:
            # Convert Claude image format to OpenAI format
            if (
                isinstance(block.source, dict)
                and block.source.get("type") == "base64"
                and "media_type" in block.source
                and "data" in block.source
            ):
                openai_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.source['media_type']};base64,{block.source['data']}"
                        },
                    }
                )

    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
        return {"role": Constants.ROLE_USER, "content": openai_content[0]["text"]}
    else:
        return {"role": Constants.ROLE_USER, "content": openai_content}


def convert_claude_assistant_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude assistant message to OpenAI format."""
    text_parts = []
    tool_calls = []

    if msg.content is None:
        return {"role": Constants.ROLE_ASSISTANT, "content": ""}
    
    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_ASSISTANT, "content": msg.content}

    for block in msg.content:
        if block.type == Constants.CONTENT_TEXT:
            text_parts.append(block.text)
        elif block.type == "compaction":
            # Compaction summary as assistant text; null content = no-op boundary
            if block.content:
                text_parts.append(block.content)
        elif block.type == Constants.CONTENT_TOOL_USE:
            tool_calls.append(
                {
                    "id": block.id,
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": block.name,
                        "arguments": json.dumps(block.input, ensure_ascii=False),
                    },
                }
            )

    openai_message = {"role": Constants.ROLE_ASSISTANT}

    # Set content
    if text_parts:
        openai_message["content"] = "".join(text_parts)
    else:
        openai_message["content"] = ""

    # Set tool calls
    if tool_calls:
        openai_message["tool_calls"] = tool_calls

    return openai_message


def convert_claude_tool_results(msg: ClaudeMessage) -> List[Dict[str, Any]]:
    """Convert Claude tool results to OpenAI format."""
    tool_messages = []

    if isinstance(msg.content, list):
        for block in msg.content:
            if block.type == Constants.CONTENT_TOOL_RESULT:
                content = parse_tool_result_content(block.content)
                tool_messages.append(
                    {
                        "role": Constants.ROLE_TOOL,
                        "tool_call_id": block.tool_use_id,
                        "content": content,
                    }
                )

    return tool_messages


def parse_tool_result_content(content):
    """Parse and normalize tool result content into a string format."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == Constants.CONTENT_TEXT:
                result_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                result_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    result_parts.append(item.get("text", ""))
                else:
                    try:
                        result_parts.append(json.dumps(item, ensure_ascii=False))
                    except:
                        result_parts.append(str(item))
        return "\n".join(result_parts).strip()

    if isinstance(content, dict):
        if content.get("type") == Constants.CONTENT_TEXT:
            return content.get("text", "")
        try:
            return json.dumps(content, ensure_ascii=False)
        except:
            return str(content)

    try:
        return str(content)
    except:
        return "Unparseable content"
