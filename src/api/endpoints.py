from fastapi import APIRouter, HTTPException, Request, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime
import uuid
import json
import time
import math
from typing import Optional

from src.core.config import config
from src.core.logging import logger
from src.core.client import OpenAIClient
from src.models.claude import ClaudeMessagesRequest, ClaudeTokenCountRequest
from src.conversion.request_converter import convert_claude_to_openai
from src.conversion.response_converter import (
    convert_openai_to_claude_response,
    convert_openai_streaming_to_claude_with_cancellation,
    convert_non_streaming_to_sse,
    convert_compaction_to_sse,
    stream_compaction_with_response,
)
from src.core.model_manager import model_manager
from src.core.compaction import (
    should_compact,
    build_compaction_messages,
    build_followup_request,
    build_compaction_response,
    build_compaction_with_response,
)

router = APIRouter()

# Get custom headers from config
custom_headers = config.get_custom_headers()

openai_client = OpenAIClient(
    config.openai_api_key,
    config.openai_base_url,
    config.request_timeout,
    api_version=config.azure_api_version,
    custom_headers=custom_headers,
)

async def validate_api_key(x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    """Validate the client's API key from either x-api-key header or Authorization header."""
    client_api_key = None
    
    # Extract API key from headers
    if x_api_key:
        client_api_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        client_api_key = authorization.replace("Bearer ", "")
    
    # Skip validation if ANTHROPIC_API_KEY is not set in the environment
    if not config.anthropic_api_key:
        return
        
    # Validate the client API key
    if not client_api_key or not config.validate_client_api_key(client_api_key):
        logger.warning(f"Invalid API key provided by client")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Please provide a valid Anthropic API key."
        )

@router.post("/v1/messages")
async def create_message(request: ClaudeMessagesRequest, http_request: Request, _: None = Depends(validate_api_key)):
    req_start = time.time()
    request_id = str(uuid.uuid4())
    short_id = request_id[:8]
    mapped_model = model_manager.map_claude_model_to_openai(request.model)
    has_tools = bool(request.tools) and len(request.tools) > 0

    logger.info(
        f"[{short_id}] >>> Request started: model={request.model}->{mapped_model}, "
        f"stream={request.stream}, has_tools={has_tools}, max_tokens={request.max_tokens}"
    )

    try:
        # Convert Claude request to OpenAI format
        convert_start = time.time()
        openai_request = convert_claude_to_openai(request, model_manager)
        convert_elapsed = time.time() - convert_start
        logger.info(f"[{short_id}] Request conversion: {convert_elapsed:.3f}s")

        # Check context window and dynamically adjust max_tokens
        input_tokens = openai_request.get("_input_tokens", 0)
        effective_input = math.ceil(input_tokens * config.token_estimate_factor)
        safety_margin = max(1024, int(effective_input * 0.05))
        max_tokens = openai_request.get("max_tokens", config.max_tokens_limit)
        context_limit = int(config.model_context_window * config.context_window_threshold)
        allowed = config.model_context_window - effective_input - safety_margin

        # Threshold warning (observe only, do not reject)
        if effective_input > context_limit:
            logger.warning(
                f"[{short_id}] Context usage high: {input_tokens}(raw)/{effective_input}(effective)/"
                f"{config.model_context_window}(window) "
                f"({effective_input/config.model_context_window*100:.1f}%, "
                f"threshold={config.context_window_threshold*100:.0f}%)"
            )

        # === Compaction check (must run BEFORE allowed<=0 rejection) ===
        compaction_edit = should_compact(request, effective_input)
        if compaction_edit:
            logger.info(f"[{short_id}] Compaction triggered, generating summary...")
            # Remove internal token count field before compaction
            openai_request.pop("_input_tokens", None)
            try:
                return await _handle_compaction(
                    request, openai_request, compaction_edit,
                    mapped_model, input_tokens, request_id, short_id,
                    req_start, http_request,
                )
            except Exception as e:
                logger.warning(f"[{short_id}] Compaction failed, falling back to normal flow: {e}")
                import traceback
                logger.warning(traceback.format_exc())
                # Fall through to normal request flow

        if allowed <= 0:
            error_response = {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        f"prompt is too long: ~{effective_input} effective tokens "
                        f"(raw={input_tokens}, factor={config.token_estimate_factor}) "
                        f"> {config.model_context_window} context window"
                    )
                }
            }
            return JSONResponse(status_code=400, content=error_response)

        # Triple clamp: min(user_max, allowed, provider_output_cap)
        final_max_tokens = max(1, min(max_tokens, allowed, config.max_output_tokens))
        if final_max_tokens != max_tokens:
            logger.warning(
                f"[{short_id}] Clamping max_tokens: {max_tokens} -> {final_max_tokens} "
                f"(input={input_tokens}, effective={effective_input}, allowed={allowed}, "
                f"output_cap={config.max_output_tokens}, window={config.model_context_window})"
            )
            openai_request["max_tokens"] = final_max_tokens

        # Remove internal token count field before sending to upstream
        openai_request.pop("_input_tokens", None)

        # Log final payload values for debugging
        logger.info(
            f"[{short_id}] Input tokens: {input_tokens}(raw)/{effective_input}(effective)/"
            f"{config.model_context_window}(window) "
            f"({effective_input/config.model_context_window*100:.1f}%), "
            f"max_tokens={openai_request.get('max_tokens')}"
        )

        # Check if client disconnected before processing
        if await http_request.is_disconnected():
            logger.info(f"[{short_id}] Client disconnected before upstream call")
            raise HTTPException(status_code=499, detail="Client disconnected")

        # Force non-streaming when tools are present (backend limitation: GLM-4.7-FP8
        # sends finish_reason='tool_calls' but doesn't include actual tool_call data in streaming)
        client_wants_streaming = request.stream
        use_streaming = request.stream and not has_tools

        if has_tools and request.stream:
            logger.info(f"[{short_id}] Tools present: forcing non-streaming backend, will convert to SSE")
            openai_request["stream"] = False

        if use_streaming:
            # Streaming response
            try:
                openai_stream = openai_client.create_chat_completion_stream(
                    openai_request, request_id
                )
                return StreamingResponse(
                    _timed_streaming_wrapper(
                        convert_openai_streaming_to_claude_with_cancellation(
                            openai_stream,
                            request,
                            logger,
                            http_request,
                            openai_client,
                            request_id,
                        ),
                        short_id,
                        req_start,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            except HTTPException as e:
                logger.error(f"[{short_id}] Streaming error: {e.detail}")
                import traceback
                logger.error(traceback.format_exc())
                error_message = openai_client.classify_openai_error(e.detail)
                error_response = {
                    "type": "error",
                    "error": {"type": "api_error", "message": error_message},
                }
                return JSONResponse(status_code=e.status_code, content=error_response)
        else:
            # Non-streaming backend call
            logger.debug(f"[{short_id}] OpenAI Request: {json.dumps(openai_request, indent=2, ensure_ascii=False)}")

            upstream_start = time.time()
            openai_response = await openai_client.create_chat_completion(
                openai_request, request_id
            )
            upstream_elapsed = time.time() - upstream_start
            logger.info(f"[{short_id}] Upstream API call: {upstream_elapsed:.3f}s")

            resp_convert_start = time.time()
            claude_response = convert_openai_to_claude_response(
                openai_response, request
            )
            resp_convert_elapsed = time.time() - resp_convert_start
            logger.info(f"[{short_id}] Response conversion: {resp_convert_elapsed:.3f}s")

            total_elapsed = time.time() - req_start
            logger.info(
                f"[{short_id}] <<< Request finished: total={total_elapsed:.3f}s "
                f"(upstream={upstream_elapsed:.3f}s, convert={convert_elapsed + resp_convert_elapsed:.3f}s)"
            )

            # If client expects streaming format, convert to SSE
            if client_wants_streaming:
                logger.info(f"[{short_id}] Converting non-streaming response to SSE format")
                return StreamingResponse(
                    convert_non_streaming_to_sse(claude_response, logger),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            else:
                return claude_response
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        total_elapsed = time.time() - req_start
        logger.error(f"[{short_id}] !!! Request failed after {total_elapsed:.3f}s: {e}")
        logger.error(traceback.format_exc())
        error_message = openai_client.classify_openai_error(str(e))
        raise HTTPException(status_code=500, detail=error_message)


async def _timed_streaming_wrapper(stream_gen, short_id: str, req_start: float):
    """Wrap a streaming generator to log timing for first chunk and total duration."""
    first_chunk_logged = False
    chunk_count = 0
    try:
        async for chunk in stream_gen:
            chunk_count += 1
            if not first_chunk_logged:
                first_chunk_time = time.time() - req_start
                logger.info(f"[{short_id}] First chunk (TTFB): {first_chunk_time:.3f}s")
                first_chunk_logged = True
            yield chunk
    finally:
        total_elapsed = time.time() - req_start
        logger.info(f"[{short_id}] <<< Stream finished: total={total_elapsed:.3f}s, chunks={chunk_count}")


async def _handle_compaction(
    request, openai_request, compaction_edit,
    mapped_model, input_tokens, request_id, short_id,
    req_start, http_request,
):
    """Handle the compaction flow: summarize conversation, optionally continue with follow-up."""
    # Determine which model to use for summarization
    compaction_model = config.compaction_model or mapped_model

    # Build summarization request
    compaction_messages = build_compaction_messages(request, compaction_edit)
    summary_request = {
        "model": compaction_model,
        "messages": compaction_messages,
        "max_tokens": config.compaction_max_tokens,
        "temperature": 0.3,
        "stream": False,
    }

    # First LLM call: generate summary
    summary_start = time.time()
    logger.info(f"[{short_id}] Compaction: sending summary request to {compaction_model}")
    summary_response = await openai_client.create_chat_completion(summary_request, request_id + "_compact")
    summary_elapsed = time.time() - summary_start

    # Extract summary text
    summary_choices = summary_response.get("choices", [])
    if not summary_choices:
        raise RuntimeError("Compaction summary returned no choices")

    summary = summary_choices[0].get("message", {}).get("content", "")
    compaction_input_tokens = summary_response.get("usage", {}).get("prompt_tokens", 0)
    compaction_output_tokens = summary_response.get("usage", {}).get("completion_tokens", 0)

    logger.info(
        f"[{short_id}] Compaction summary generated: {summary_elapsed:.3f}s, "
        f"input={compaction_input_tokens}, output={compaction_output_tokens}, "
        f"summary_len={len(summary)}"
    )

    compaction_usage = {
        "input_tokens": compaction_input_tokens,
        "output_tokens": compaction_output_tokens,
    }

    # pause_after_compaction=true: just return the compaction block
    if compaction_edit.pause_after_compaction:
        claude_response = build_compaction_response(
            summary, request, compaction_input_tokens, compaction_output_tokens
        )
        total_elapsed = time.time() - req_start
        logger.info(f"[{short_id}] <<< Compaction (pause) finished: total={total_elapsed:.3f}s")

        if request.stream:
            return StreamingResponse(
                _timed_streaming_wrapper(
                    convert_compaction_to_sse(claude_response, logger),
                    short_id, req_start,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )
        else:
            return claude_response

    # pause_after_compaction=false: build follow-up request and continue
    followup_request = build_followup_request(summary, request, mapped_model, openai_request)
    has_tools = bool(request.tools) and len(request.tools) > 0
    client_wants_streaming = request.stream

    logger.info(f"[{short_id}] Compaction: sending follow-up request (stream={client_wants_streaming})")

    if client_wants_streaming and not has_tools:
        # Streaming follow-up
        followup_request["stream"] = True
        openai_stream = openai_client.create_chat_completion_stream(followup_request, request_id + "_followup")

        return StreamingResponse(
            _timed_streaming_wrapper(
                stream_compaction_with_response(
                    summary, openai_stream, request, logger,
                    http_request, openai_client, request_id + "_followup",
                    compaction_usage,
                ),
                short_id, req_start,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )
    else:
        # Non-streaming follow-up
        followup_request["stream"] = False
        followup_start = time.time()
        followup_response = await openai_client.create_chat_completion(
            followup_request, request_id + "_followup"
        )
        followup_elapsed = time.time() - followup_start
        logger.info(f"[{short_id}] Compaction follow-up: {followup_elapsed:.3f}s")

        # Convert follow-up response to Claude format
        claude_followup = convert_openai_to_claude_response(followup_response, request)

        # Build combined response
        message_input_tokens = followup_response.get("usage", {}).get("prompt_tokens", 0)
        message_output_tokens = followup_response.get("usage", {}).get("completion_tokens", 0)

        combined_response = build_compaction_with_response(
            summary,
            claude_followup.get("content", []),
            request,
            claude_followup.get("stop_reason", Constants.STOP_END_TURN),
            compaction_input_tokens, compaction_output_tokens,
            message_input_tokens, message_output_tokens,
        )

        total_elapsed = time.time() - req_start
        logger.info(f"[{short_id}] <<< Compaction+response finished: total={total_elapsed:.3f}s")

        if client_wants_streaming:
            return StreamingResponse(
                convert_non_streaming_to_sse(combined_response, logger),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )
        else:
            return combined_response


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: ClaudeTokenCountRequest, _: None = Depends(validate_api_key)):
    try:
        # For token counting, we'll use a simple estimation

        total_chars = 0

        # Count system message characters
        if request.system:
            if isinstance(request.system, str):
                total_chars += len(request.system)
            elif isinstance(request.system, list):
                for block in request.system:
                    if hasattr(block, "text"):
                        total_chars += len(block.text)

        # Count message characters
        for msg in request.messages:
            if msg.content is None:
                continue
            elif isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text") and block.text is not None:
                        total_chars += len(block.text)

        # Rough estimation: 4 characters per token
        estimated_tokens = max(1, total_chars // 4)

        return {"input_tokens": estimated_tokens}

    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_api_configured": bool(config.openai_api_key),
        "api_key_valid": config.validate_api_key(),
        "client_api_key_validation": bool(config.anthropic_api_key),
    }


@router.get("/test-connection")
async def test_connection():
    """Test API connectivity to OpenAI"""
    try:
        # Simple test request to verify API connectivity
        test_response = await openai_client.create_chat_completion(
            {
                "model": config.small_model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            }
        )

        return {
            "status": "success",
            "message": "Successfully connected to OpenAI API",
            "model_used": config.small_model,
            "timestamp": datetime.now().isoformat(),
            "response_id": test_response.get("id", "unknown"),
        }

    except Exception as e:
        logger.error(f"API connectivity test failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "error_type": "API Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
                "suggestions": [
                    "Check your OPENAI_API_KEY is valid",
                    "Verify your API key has the necessary permissions",
                    "Check if you have reached rate limits",
                ],
            },
        )


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Claude-to-OpenAI API Proxy v1.0.0",
        "status": "running",
        "config": {
            "openai_base_url": config.openai_base_url,
            "max_tokens_limit": config.max_tokens_limit,
            "api_key_configured": bool(config.openai_api_key),
            "client_api_key_validation": bool(config.anthropic_api_key),
            "big_model": config.big_model,
            "small_model": config.small_model,
        },
        "endpoints": {
            "messages": "/v1/messages",
            "count_tokens": "/v1/messages/count_tokens",
            "health": "/health",
            "test_connection": "/test-connection",
        },
    }
