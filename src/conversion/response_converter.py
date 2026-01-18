import json
import re
import uuid
from fastapi import HTTPException, Request
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest


def parse_glm_react_format(content: str) -> list:
    """Parse GLM-4.7 ReAct format tool calls from content.

    Format: <tool_call>function_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>

    Returns list of tool call dicts with 'id', 'name', and 'arguments' keys.
    """
    tool_calls = []
    logger = logging.getLogger(__name__)

    logger.debug(f"=== Parsing ReAct Format ===")
    logger.debug(f"Content length: {len(content)}")
    logger.debug(f"Content preview (first 500 chars): {content[:500]}")

    # Find all <tool_call>...</tool_call> blocks (case-insensitive, allow whitespace)
    tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(tool_call_pattern, content, re.DOTALL | re.IGNORECASE)

    logger.debug(f"Found {len(matches)} tool_call blocks")

    for i, match in enumerate(matches):
        try:
            logger.debug(f"Processing tool_call block {i}: {match[:200]}...")

            # Extract function name - stop at first < or whitespace
            # This handles cases where function name is immediately followed by <arg_key>
            name_match = re.search(r'^([a-zA-Z_][a-zA-Z0-9_]*)(?=\s|<|$)', match.strip())
            if not name_match:
                logger.warning(f"Could not extract function name from tool_call block {i}")
                logger.debug(f"Block content: {match}")
                continue

            function_name = name_match.group(1).strip()
            logger.debug(f"Extracted function name: {function_name}")

            # Extract all arg_key/arg_value pairs (allow whitespace, case-insensitive)
            arg_pattern = r'<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>'
            args = re.findall(arg_pattern, match, re.DOTALL | re.IGNORECASE)

            logger.debug(f"Found {len(args)} argument pairs")

            # Build arguments dict
            arguments = {}
            for j, (key, value) in enumerate(args):
                key = key.strip()
                value = value.strip()

                logger.debug(f"Arg {j}: key='{key}', value_length={len(value)}")

                # Try to parse value as JSON (for complex types like arrays/objects)
                try:
                    parsed_value = json.loads(value)
                    arguments[key] = parsed_value
                    logger.debug(f"Successfully parsed arg '{key}' as JSON")
                except json.JSONDecodeError:
                    # Keep as string if not valid JSON
                    arguments[key] = value
                    logger.debug(f"Keeping arg '{key}' as string")

            logger.debug(f"Final arguments: {json.dumps(arguments, ensure_ascii=False)[:200]}")

            tool_calls.append({
                'id': f'toolu_{uuid.uuid4().hex[:24]}',
                'type': 'function',
                'function': {
                    'name': function_name,
                    'arguments': json.dumps(arguments, ensure_ascii=False)
                }
            })

            logger.debug(f"Successfully parsed tool call: {function_name}")

        except Exception as e:
            logger.error(f"Error parsing tool_call block {i}: {e}")
            logger.error(f"Block content: {match[:500]}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    logger.debug(f"=== ReAct Parsing Complete: {len(tool_calls)} tool calls ===")
    return tool_calls


import logging

response_logger = logging.getLogger(__name__)

def convert_openai_to_claude_response(
    openai_response: dict, original_request: ClaudeMessagesRequest
) -> dict:
    """Convert OpenAI response to Claude format."""

    response_logger.debug(f"=== Converting OpenAI Response to Claude Format ===")
    response_logger.debug(f"OpenAI Response: {json.dumps(openai_response, indent=2, ensure_ascii=False)}")

    # Extract response data
    choices = openai_response.get("choices", [])
    if not choices:
        raise HTTPException(status_code=500, detail="No choices in OpenAI response")

    choice = choices[0]
    message = choice.get("message", {})

    # Build Claude content blocks
    content_blocks = []

    # Check for GLM-4.7 ReAct format in content
    text_content = message.get("content")
    tool_calls = message.get("tool_calls", []) or []

    response_logger.debug(f"=== Tool Call Detection ===")
    response_logger.debug(f"text_content length: {len(text_content) if text_content else 0}")
    response_logger.debug(f"tool_calls count (from OpenAI): {len(tool_calls)}")
    response_logger.debug(f"Has <tool_call> tag: {'<tool_call>' in text_content if text_content else False}")

    # Parse ReAct format if tool_calls is empty but content has <tool_call> tags
    if text_content and not tool_calls and '<tool_call>' in text_content:
        response_logger.debug(f"Attempting ReAct format parsing...")
        response_logger.debug(f"Content preview: {text_content[:500]}")

        react_tool_calls = parse_glm_react_format(text_content)

        response_logger.debug(f"Parsed {len(react_tool_calls)} tool calls from ReAct format")
        tool_calls.extend(react_tool_calls)

        # Clean content: remove <think> and <tool_call> tags
        cleaned_content = re.sub(r'<think>.*?</think>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
        cleaned_content = re.sub(r'</think>', '', cleaned_content, flags=re.IGNORECASE)  # Remove standalone closing tags
        cleaned_content = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned_content, flags=re.DOTALL | re.IGNORECASE)
        text_content = cleaned_content.strip()

        response_logger.debug(f"Cleaned text_content length: {len(text_content)}")

    # Add text content
    if text_content:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": text_content})

    # Add tool calls
    for tool_call in tool_calls:
        if tool_call.get("type") == Constants.TOOL_FUNCTION:
            function_data = tool_call.get(Constants.TOOL_FUNCTION, {})
            try:
                arguments = json.loads(function_data.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {"raw_arguments": function_data.get("arguments", "")}

            # Generate Claude-style tool call ID (must start with toolu_)
            tool_id = tool_call.get("id", "")
            if tool_id and not tool_id.startswith("toolu_"):
                # Convert to Claude format
                tool_id = f"toolu_{tool_id.replace('call_', '').replace('tool_', '')}"
            else:
                tool_id = tool_id if tool_id else f"toolu_{uuid.uuid4().hex[:24]}"

            content_blocks.append(
                {
                    "type": Constants.CONTENT_TOOL_USE,
                    "id": tool_id,
                    "name": function_data.get("name", ""),
                    "input": arguments,
                }
            )

    # Ensure at least one content block
    if not content_blocks:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": ""})

    # Map finish reason
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = {
        "stop": Constants.STOP_END_TURN,
        "length": Constants.STOP_MAX_TOKENS,
        "tool_calls": Constants.STOP_TOOL_USE,
        "function_call": Constants.STOP_TOOL_USE,
    }.get(finish_reason, Constants.STOP_END_TURN)

    # Override stop_reason if we have tool_use content blocks (from ReAct format or standard format)
    has_tool_use = any(block.get("type") == Constants.CONTENT_TOOL_USE for block in content_blocks)
    if has_tool_use:
        stop_reason = Constants.STOP_TOOL_USE

    # Generate Claude-style message ID (must start with msg_)
    openai_id = openai_response.get("id", "")
    if openai_id and not openai_id.startswith("msg_"):
        # Convert OpenAI ID format (chatcmpl-xxx) to Claude format (msg_xxx)
        message_id = f"msg_{openai_id.replace('chatcmpl-', '')}"
    else:
        message_id = openai_id if openai_id else f"msg_{uuid.uuid4().hex[:24]}"

    # Build Claude response
    claude_response = {
        "id": message_id,
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_response.get("usage", {}).get(
                "completion_tokens", 0
            ),
        },
    }

    response_logger.debug(f"=== Final Claude Response ===")
    response_logger.debug(f"Claude Response: {json.dumps(claude_response, indent=2, ensure_ascii=False)}")
    response_logger.debug(f"stop_reason: {stop_reason}, has_tool_use: {has_tool_use}")

    return claude_response


async def convert_openai_streaming_to_claude(
    openai_stream, original_request: ClaudeMessagesRequest, logger
):
    """Convert OpenAI streaming response to Claude streaming format."""

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Send initial SSE events
    yield f"event: {Constants.EVENT_MESSAGE_START}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_START, 'message': {'id': message_id, 'type': 'message', 'role': Constants.ROLE_ASSISTANT, 'model': original_request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': 0, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_PING}\ndata: {json.dumps({'type': Constants.EVENT_PING}, ensure_ascii=False)}\n\n"

    # Process streaming chunks
    text_block_index = 0
    tool_block_counter = 0
    current_tool_calls = {}
    final_stop_reason = Constants.STOP_END_TURN

    try:
        async for line in openai_stream:
            if line.strip():
                if line.startswith("data: "):
                    chunk_data = line[6:]
                    if chunk_data.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(chunk_data)
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to parse chunk: {chunk_data}, error: {e}"
                        )
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Handle text delta
                    if delta and "content" in delta and delta["content"] is not None:
                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': text_block_index, 'delta': {'type': Constants.DELTA_TEXT, 'text': delta['content']}}, ensure_ascii=False)}\n\n"

                    # Handle tool call deltas with improved incremental processing
                    if "tool_calls" in delta:
                        for tc_delta in delta["tool_calls"]:
                            tc_index = tc_delta.get("index", 0)
                            
                            # Initialize tool call tracking by index if not exists
                            if tc_index not in current_tool_calls:
                                current_tool_calls[tc_index] = {
                                    "id": None,
                                    "name": None,
                                    "args_buffer": "",
                                    "json_sent": False,
                                    "claude_index": None,
                                    "started": False
                                }
                            
                            tool_call = current_tool_calls[tc_index]
                            
                            # Update tool call ID if provided
                            if tc_delta.get("id"):
                                tool_call["id"] = tc_delta["id"]
                            
                            # Update function name and start content block if we have both id and name
                            function_data = tc_delta.get(Constants.TOOL_FUNCTION, {})
                            if function_data.get("name"):
                                tool_call["name"] = function_data["name"]
                            
                            # Start content block when we have complete initial data
                            if (tool_call["id"] and tool_call["name"] and not tool_call["started"]):
                                tool_block_counter += 1
                                claude_index = text_block_index + tool_block_counter
                                tool_call["claude_index"] = claude_index
                                tool_call["started"] = True
                                
                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': claude_index, 'content_block': {'type': Constants.CONTENT_TOOL_USE, 'id': tool_call['id'], 'name': tool_call['name'], 'input': {}}}, ensure_ascii=False)}\n\n"
                            
                            # Handle function arguments
                            if "arguments" in function_data and tool_call["started"] and function_data["arguments"] is not None:
                                tool_call["args_buffer"] += function_data["arguments"]
                                
                                # Try to parse complete JSON and send delta when we have valid JSON
                                try:
                                    json.loads(tool_call["args_buffer"])
                                    # Send the current complete JSON state
                                    yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': tool_call['claude_index'], 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': tool_call['args_buffer']}}, ensure_ascii=False)}\n\n"
                                except json.JSONDecodeError:
                                    # JSON is incomplete, continue accumulating
                                    pass

                    # Handle finish reason
                    if finish_reason:
                        if finish_reason == "length":
                            final_stop_reason = Constants.STOP_MAX_TOKENS
                        elif finish_reason in ["tool_calls", "function_call"]:
                            final_stop_reason = Constants.STOP_TOOL_USE
                        elif finish_reason == "stop":
                            final_stop_reason = Constants.STOP_END_TURN
                        else:
                            final_stop_reason = Constants.STOP_END_TURN
                        break

    except Exception as e:
        # Handle any streaming errors gracefully
        logger.error(f"Streaming error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": f"Streaming error: {str(e)}"},
        }
        yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        return

    # Send final SSE events
    yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': text_block_index}, ensure_ascii=False)}\n\n"

    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("claude_index") is not None:
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': tool_data['claude_index']}, ensure_ascii=False)}\n\n"

    usage_data = {"input_tokens": 0, "output_tokens": 0}
    yield f"event: {Constants.EVENT_MESSAGE_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_DELTA, 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': usage_data}, ensure_ascii=False)}\n\n"
    yield f"event: {Constants.EVENT_MESSAGE_STOP}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_STOP}, ensure_ascii=False)}\n\n"


async def convert_openai_streaming_to_claude_with_cancellation(
    openai_stream,
    original_request: ClaudeMessagesRequest,
    logger,
    http_request: Request,
    openai_client,
    request_id: str,
):
    """Convert OpenAI streaming response to Claude streaming format with cancellation support."""

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Send initial SSE events
    yield f"event: {Constants.EVENT_MESSAGE_START}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_START, 'message': {'id': message_id, 'type': 'message', 'role': Constants.ROLE_ASSISTANT, 'model': original_request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': 0, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_PING}\ndata: {json.dumps({'type': Constants.EVENT_PING}, ensure_ascii=False)}\n\n"

    # Process streaming chunks
    text_block_index = 0
    tool_block_counter = 0
    current_tool_calls = {}
    final_stop_reason = Constants.STOP_END_TURN
    usage_data = {"input_tokens": 0, "output_tokens": 0}

    # ReAct format detection
    content_buffer = ""
    is_react_format = False
    react_text_sent = False

    try:
        async for line in openai_stream:
            # Check if client disconnected
            if await http_request.is_disconnected():
                logger.info(f"Client disconnected, cancelling request {request_id}")
                openai_client.cancel_request(request_id)
                break

            if line.strip():
                if line.startswith("data: "):
                    chunk_data = line[6:]
                    if chunk_data.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(chunk_data)
                        # logger.info(f"OpenAI chunk: {chunk}")
                        usage = chunk.get("usage", None)
                        if usage:
                            cache_read_input_tokens = 0
                            prompt_tokens_details = usage.get('prompt_tokens_details', {})
                            if prompt_tokens_details:
                                cache_read_input_tokens = prompt_tokens_details.get('cached_tokens', 0)
                            usage_data = {
                                'input_tokens': usage.get('prompt_tokens', 0),
                                'output_tokens': usage.get('completion_tokens', 0),
                                'cache_read_input_tokens': cache_read_input_tokens
                            }
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to parse chunk: {chunk_data}, error: {e}"
                        )
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Debug: log the raw delta content
                    if delta:
                        logger.debug(f"Raw delta content: {delta}")

                    # Handle text delta
                    if delta and "content" in delta and delta["content"] is not None:
                        content_buffer += delta["content"]

                        # Detect ReAct format
                        if not is_react_format and ('<think>' in content_buffer or '<tool_call>' in content_buffer):
                            is_react_format = True

                        # If not ReAct format, send text delta immediately
                        if not is_react_format:
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': text_block_index, 'delta': {'type': Constants.DELTA_TEXT, 'text': delta['content']}}, ensure_ascii=False)}\n\n"

                    # Handle tool call deltas with improved incremental processing
                    if "tool_calls" in delta and delta["tool_calls"]:
                        logger.debug(f"Received tool_calls delta: {delta['tool_calls']}")
                        for tc_delta in delta["tool_calls"]:
                            tc_index = tc_delta.get("index", 0)
                            
                            # Initialize tool call tracking by index if not exists
                            if tc_index not in current_tool_calls:
                                current_tool_calls[tc_index] = {
                                    "id": None,
                                    "name": None,
                                    "args_buffer": "",
                                    "json_sent": False,
                                    "claude_index": None,
                                    "started": False
                                }
                            
                            tool_call = current_tool_calls[tc_index]
                            
                            # Update tool call ID if provided
                            if tc_delta.get("id"):
                                tool_call["id"] = tc_delta["id"]
                            
                            # Update function name and start content block if we have both id and name
                            function_data = tc_delta.get(Constants.TOOL_FUNCTION, {})
                            if function_data.get("name"):
                                tool_call["name"] = function_data["name"]
                            
                            # Start content block when we have complete initial data
                            if (tool_call["id"] and tool_call["name"] and not tool_call["started"]):
                                tool_block_counter += 1
                                claude_index = text_block_index + tool_block_counter
                                tool_call["claude_index"] = claude_index
                                tool_call["started"] = True
                                
                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': claude_index, 'content_block': {'type': Constants.CONTENT_TOOL_USE, 'id': tool_call['id'], 'name': tool_call['name'], 'input': {}}}, ensure_ascii=False)}\n\n"
                            
                            # Handle function arguments
                            if "arguments" in function_data and tool_call["started"] and function_data["arguments"] is not None:
                                tool_call["args_buffer"] += function_data["arguments"]
                                
                                # Try to parse complete JSON and send delta when we have valid JSON
                                try:
                                    json.loads(tool_call["args_buffer"])
                                    # Send the current complete JSON state
                                    yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': tool_call['claude_index'], 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': tool_call['args_buffer']}}, ensure_ascii=False)}\n\n"
                                except json.JSONDecodeError:
                                    # JSON is incomplete, continue accumulating
                                    pass

                    # Handle finish reason
                    if finish_reason:
                        logger.debug(f"OpenAI finish_reason: {finish_reason}")
                        logger.debug(f"Final chunk when finish_reason received: {chunk}")
                        if finish_reason == "length":
                            final_stop_reason = Constants.STOP_MAX_TOKENS
                        elif finish_reason in ["tool_calls", "function_call"]:
                            final_stop_reason = Constants.STOP_TOOL_USE
                            logger.debug(f"Setting stop_reason to 'tool_use' based on finish_reason: {finish_reason}")
                        elif finish_reason == "stop":
                            final_stop_reason = Constants.STOP_END_TURN
                        else:
                            final_stop_reason = Constants.STOP_END_TURN

        # Process ReAct format content after stream ends
        logger.debug(f"=== Stream End: ReAct Format Processing ===")
        logger.debug(f"is_react_format={is_react_format}, content_buffer_len={len(content_buffer) if content_buffer else 0}")
        if content_buffer:
            logger.debug(f"content_buffer contains <tool_call>: {'<tool_call>' in content_buffer}")
            logger.debug(f"content_buffer contains <think>: {'<think>' in content_buffer}")
            logger.debug(f"content_buffer preview (last 500 chars): {content_buffer[-500:] if len(content_buffer) > 500 else content_buffer}")

        if is_react_format and content_buffer:
            logger.debug(f"Processing ReAct format content from stream...")

            # Parse ReAct format tool calls
            react_tool_calls = parse_glm_react_format(content_buffer)
            logger.debug(f"Parsed {len(react_tool_calls)} tool calls from ReAct format content")

            # Clean content: remove <think> and <tool_call> tags (case-insensitive)
            cleaned_content = re.sub(r'<think>.*?</think>', '', content_buffer, flags=re.DOTALL | re.IGNORECASE)
            cleaned_content = re.sub(r'</think>', '', cleaned_content, flags=re.IGNORECASE)  # Remove standalone closing tags
            cleaned_content = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned_content, flags=re.DOTALL | re.IGNORECASE)
            cleaned_content = cleaned_content.strip()

            logger.debug(f"Cleaned content length: {len(cleaned_content)}")

            # Send cleaned text if any
            if cleaned_content and not react_text_sent:
                yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': text_block_index, 'delta': {'type': Constants.DELTA_TEXT, 'text': cleaned_content}}, ensure_ascii=False)}\n\n"
                react_text_sent = True

            # Send tool calls
            for tool_call in react_tool_calls:
                tool_block_counter += 1
                claude_index = text_block_index + tool_block_counter

                function_data = tool_call.get('function', {})
                tool_id = tool_call.get('id', f'toolu_{uuid.uuid4().hex[:24]}')
                tool_name = function_data.get('name', '')

                # Parse arguments
                try:
                    arguments = json.loads(function_data.get('arguments', ''))
                except json.JSONDecodeError:
                    arguments = {}

                # Send tool call start event
                yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': claude_index, 'content_block': {'type': Constants.CONTENT_TOOL_USE, 'id': tool_id, 'name': tool_name, 'input': {}}}, ensure_ascii=False)}\n\n"

                # Send tool call delta with complete arguments
                if arguments:
                    yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': claude_index, 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': json.dumps(arguments, ensure_ascii=False)}}, ensure_ascii=False)}\n\n"

                # Send tool call stop event
                yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': claude_index}, ensure_ascii=False)}\n\n"

            # Update stop reason if we sent tool calls (moved outside the loop)
            if react_tool_calls:
                final_stop_reason = Constants.STOP_TOOL_USE
                logger.debug(f"ReAct format: parsed {len(react_tool_calls)} tool calls, setting stop_reason to 'tool_use'")

    except HTTPException as e:
        # Handle cancellation
        if e.status_code == 499:
            logger.info(f"Request {request_id} was cancelled")
            error_event = {
                "type": "error",
                "error": {
                    "type": "cancelled",
                    "message": "Request was cancelled by client",
                },
            }
            yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            return
        else:
            raise
    except Exception as e:
        # Handle any streaming errors gracefully
        logger.error(f"Streaming error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": f"Streaming error: {str(e)}"},
        }
        yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        return

    # Send final SSE events
    yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': text_block_index}, ensure_ascii=False)}\n\n"

    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("claude_index") is not None:
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': tool_data['claude_index']}, ensure_ascii=False)}\n\n"

    logger.debug(f"Sending message_delta with stop_reason: {final_stop_reason}, current_tool_calls count: {len(current_tool_calls)}")
    yield f"event: {Constants.EVENT_MESSAGE_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_DELTA, 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': usage_data}, ensure_ascii=False)}\n\n"
    yield f"event: {Constants.EVENT_MESSAGE_STOP}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_STOP}, ensure_ascii=False)}\n\n"


async def convert_non_streaming_to_sse(claude_response: dict, logger):
    """Convert a complete Claude response (JSON) to SSE streaming format.

    This is used when the backend needs non-streaming (e.g., for tool calls),
    but the client expects streaming format.
    """

    message_id = claude_response.get('id', f'msg_{uuid.uuid4().hex[:24]}')
    model = claude_response.get('model', '')
    content_blocks = claude_response.get('content', [])
    stop_reason = claude_response.get('stop_reason', Constants.STOP_END_TURN)
    usage = claude_response.get('usage', {'input_tokens': 0, 'output_tokens': 0})

    logger.debug(f"Converting non-streaming response to SSE format: {len(content_blocks)} content blocks")

    # 1. Send message_start event
    yield f"event: {Constants.EVENT_MESSAGE_START}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_START, 'message': {'id': message_id, 'type': 'message', 'role': Constants.ROLE_ASSISTANT, 'model': model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}}, ensure_ascii=False)}\n\n"

    # 2. Send ping event
    yield f"event: {Constants.EVENT_PING}\ndata: {json.dumps({'type': Constants.EVENT_PING}, ensure_ascii=False)}\n\n"

    # 3. Process each content block
    for index, block in enumerate(content_blocks):
        block_type = block.get('type')

        if block_type == Constants.CONTENT_TEXT:
            # Text content block
            text = block.get('text', '')

            # Send content_block_start
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': index, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}}, ensure_ascii=False)}\n\n"

            # Send content_block_delta with text
            if text:
                yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': index, 'delta': {'type': Constants.DELTA_TEXT, 'text': text}}, ensure_ascii=False)}\n\n"

            # Send content_block_stop
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': index}, ensure_ascii=False)}\n\n"

        elif block_type == Constants.CONTENT_TOOL_USE:
            # Tool use content block
            tool_id = block.get('id', f'toolu_{uuid.uuid4().hex[:24]}')
            tool_name = block.get('name', '')
            tool_input = block.get('input', {})

            # Send content_block_start
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': index, 'content_block': {'type': Constants.CONTENT_TOOL_USE, 'id': tool_id, 'name': tool_name, 'input': {}}}, ensure_ascii=False)}\n\n"

            # Send content_block_delta with tool input
            if tool_input:
                yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': index, 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': json.dumps(tool_input, ensure_ascii=False)}}, ensure_ascii=False)}\n\n"

            # Send content_block_stop
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': index}, ensure_ascii=False)}\n\n"

    # 4. Send message_delta with stop_reason and usage
    yield f"event: {Constants.EVENT_MESSAGE_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_DELTA, 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage}, ensure_ascii=False)}\n\n"

    # 5. Send message_stop
    yield f"event: {Constants.EVENT_MESSAGE_STOP}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_STOP}, ensure_ascii=False)}\n\n"

    logger.debug(f"SSE conversion complete: stop_reason={stop_reason}")
