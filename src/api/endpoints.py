from fastapi import APIRouter, HTTPException, Request, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime
import uuid
import json
import time
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
)
from src.core.model_manager import model_manager

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
        safety_margin = 256
        max_tokens = openai_request.get("max_tokens", config.max_tokens_limit)
        context_limit = int(config.model_context_window * config.context_window_threshold)
        allowed = config.model_context_window - input_tokens - safety_margin

        # Threshold warning (observe only, do not reject)
        if input_tokens > context_limit:
            logger.warning(
                f"[{short_id}] Context usage high: {input_tokens}/{config.model_context_window} tokens "
                f"({input_tokens/config.model_context_window*100:.1f}%, threshold={config.context_window_threshold*100:.0f}%)"
            )

        if allowed <= 0:
            # Input already exceeds window, reject immediately
            error_response = {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"prompt is too long: {input_tokens} tokens > {config.model_context_window} maximum"
                }
            }
            return JSONResponse(status_code=400, content=error_response)

        # Dynamically clamp max_tokens to fit within context window
        if max_tokens > allowed:
            logger.warning(
                f"[{short_id}] Clamping max_tokens: {max_tokens} -> {allowed} "
                f"(input={input_tokens}, window={config.model_context_window}, margin={safety_margin})"
            )
            openai_request["max_tokens"] = max(1, allowed)

        # Remove internal token count field before sending to upstream
        openai_request.pop("_input_tokens", None)

        logger.info(f"[{short_id}] Input tokens: {input_tokens}/{config.model_context_window} ({input_tokens/config.model_context_window*100:.1f}%)")

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
