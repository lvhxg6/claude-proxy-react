# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Claude Code Proxy is a Python server that enables Claude Code CLI to work with OpenAI-compatible API providers. It converts Claude API requests to OpenAI format, allowing use of various LLM providers (OpenAI, Azure OpenAI, Ollama, etc.) through the Claude Code interface.

## Common Commands

```bash
# Install dependencies
uv sync

# Run the proxy server
uv run claude-code-proxy
# or
python start_proxy.py

# Format code
uv run black src/
uv run isort src/

# Type checking
uv run mypy src/

# Run tests
python tests/test_main.py
python src/test_claude_to_openai.py
```

## Architecture

### Request Flow
```
Claude Code CLI → Proxy (FastAPI) → Request Converter → OpenAI Client → Target LLM API
                                                                              ↓
Claude Code CLI ← Proxy            ← Response Converter ←──────────────────────
```

### Core Components

**`src/core/`** - Foundation layer:
- `config.py` - Environment variable management, validates `OPENAI_API_KEY`, model mapping (`BIG_MODEL`/`MIDDLE_MODEL`/`SMALL_MODEL`), custom headers
- `client.py` - Async OpenAI client wrapper with streaming, cancellation support, and error classification
- `model_manager.py` - Maps Claude model names to configured OpenAI models (haiku→SMALL, sonnet→MIDDLE, opus→BIG)

**`src/conversion/`** - Protocol translation:
- `request_converter.py` - Converts Claude messages format to OpenAI chat completion format (handles system messages, multimodal content, tool definitions)
- `response_converter.py` - Converts OpenAI responses back to Claude format (both streaming SSE and non-streaming)

**`src/api/endpoints.py`** - FastAPI routes:
- `POST /v1/messages` - Main endpoint (streaming & non-streaming)
- `POST /v1/messages/count_tokens` - Token estimation
- `GET /health`, `GET /test-connection` - Health checks

**`src/models/claude.py`** - Pydantic models for Claude API request/response validation

### Key Patterns

- **Async throughout**: All API calls use `asyncio` for non-blocking I/O
- **Request cancellation**: Tracks active requests by ID via `asyncio.Event` for cleanup on client disconnect
- **Streaming**: Uses Server-Sent Events (SSE) format with incremental tool call JSON parsing
- **Custom headers**: Environment variables prefixed with `CUSTOM_HEADER_` are injected into all outgoing requests

## Configuration

Required: `OPENAI_API_KEY`

Model mapping (environment variables):
- `BIG_MODEL` (default: `gpt-4o`) - For Claude opus
- `MIDDLE_MODEL` (default: same as BIG_MODEL) - For Claude sonnet
- `SMALL_MODEL` (default: `gpt-4o-mini`) - For Claude haiku

Server: `HOST`, `PORT`, `LOG_LEVEL`

## Commit Message Rules

When generating commits:
- Do NOT include `Co-Authored-By: Claude <noreply@anthropic.com>`
- Do NOT include `Generated with Claude Code` attribution
