# GLM-4.7 Integration Guide

## Overview

GLM-4.7-FP8 is a quantized version of the GLM-4 model that uses a non-standard **ReAct format** for tool calls instead of OpenAI's standard function calling format. This guide explains how the proxy handles this format and how to optimize your integration.

## ReAct Format vs OpenAI Function Calling

### Standard OpenAI Format

```json
{
  "message": {
    "content": "I'll check the weather for you.",
    "tool_calls": [
      {
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"San Francisco\"}"
        }
      }
    ]
  },
  "finish_reason": "tool_calls"
}
```

### GLM-4.7 ReAct Format

```json
{
  "message": {
    "content": "<think>The user wants weather info. I should use get_weather.</think>I'll check the weather for you.<tool_call>get_weather<arg_key>location</arg_key><arg_value>San Francisco</arg_value></tool_call>",
    "tool_calls": []
  },
  "finish_reason": "stop"
}
```

**Key differences:**
1. Tool calls are embedded in the `content` field as XML-like tags
2. `tool_calls` array is empty
3. `finish_reason` is `"stop"` instead of `"tool_calls"`
4. Includes `<think>` tags for reasoning (which should be removed)

## How the Proxy Handles ReAct Format

### Detection

The proxy detects ReAct format when:
1. `tool_calls` array is empty or null
2. `content` contains `<tool_call>` tags

```python
if text_content and not tool_calls and '<tool_call>' in text_content:
    # Parse ReAct format
    react_tool_calls = parse_glm_react_format(text_content)
```

### Parsing

The `parse_glm_react_format` function:

1. **Finds tool call blocks** using regex:
   ```python
   tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
   matches = re.findall(tool_call_pattern, content, re.DOTALL | re.IGNORECASE)
   ```

2. **Extracts function name**:
   ```python
   name_match = re.search(r'^([a-zA-Z_][a-zA-Z0-9_]*)', match.strip())
   function_name = name_match.group(1).strip()
   ```

3. **Extracts arguments**:
   ```python
   arg_pattern = r'<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>'
   args = re.findall(arg_pattern, match, re.DOTALL | re.IGNORECASE)
   ```

4. **Parses JSON values** (for complex types):
   ```python
   try:
       parsed_value = json.loads(value)
       arguments[key] = parsed_value
   except json.JSONDecodeError:
       arguments[key] = value  # Keep as string
   ```

5. **Converts to OpenAI format**:
   ```python
   {
       'id': f'toolu_{uuid.uuid4().hex[:24]}',
       'type': 'function',
       'function': {
           'name': function_name,
           'arguments': json.dumps(arguments, ensure_ascii=False)
       }
   }
   ```

### Cleaning

After parsing, the proxy removes ReAct-specific tags:

```python
# Remove <think> tags
cleaned_content = re.sub(r'<think>.*?</think>', '', text_content, flags=re.DOTALL | re.IGNORECASE)

# Remove <tool_call> tags
cleaned_content = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned_content, flags=re.DOTALL | re.IGNORECASE)
```

### Stop Reason Override

The proxy overrides the stop reason when tool calls are detected:

```python
has_tool_use = any(block.get("type") == Constants.CONTENT_TOOL_USE for block in content_blocks)
if has_tool_use:
    stop_reason = Constants.STOP_TOOL_USE  # "tool_use"
```

## Configuration for GLM-4.7

### Basic Setup

```bash
# .env
OPENAI_API_KEY="your-glm-api-key"
OPENAI_BASE_URL="http://your-glm-server:8888/v1"

# All Claude models map to GLM-4.7-FP8
BIG_MODEL="GLM-4.7-FP8"
MIDDLE_MODEL="GLM-4.7-FP8"
SMALL_MODEL="GLM-4.7-FP8"

# Enable debug logging to monitor ReAct parsing
LOG_LEVEL="DEBUG"

# GLM-4.7 supports 200K context, 128K output
MAX_TOKENS_LIMIT="65536"
REQUEST_TIMEOUT="120"
```

### Streaming vs Non-Streaming

**Important:** When tools are present, the proxy automatically uses non-streaming mode for the backend call, but still returns SSE format to the client.

This is because GLM-4.7 has issues with streaming tool calls:
- May send `finish_reason='tool_calls'` without actual tool call data
- ReAct format parsing is more reliable in non-streaming mode

```python
# In endpoints.py
has_tools = bool(request.tools) and len(request.tools) > 0
use_streaming = request.stream and not has_tools

if has_tools and request.stream:
    logger.debug("Tools present: using non-streaming backend but returning SSE format to client")
    openai_request["stream"] = False
```

## Monitoring and Debugging

### Enable Detailed Logging

```bash
LOG_LEVEL="DEBUG"
```

### Key Log Messages

**Successful parsing:**
```
=== Parsing ReAct Format ===
Content length: 1234
Found 1 tool_call blocks
Processing tool_call block 0: TodoWrite...
Extracted function name: TodoWrite
Found 1 argument pairs
Successfully parsed arg 'todos' as JSON
Successfully parsed tool call: TodoWrite
=== ReAct Parsing Complete: 1 tool calls ===
```

**Failed parsing:**
```
Could not extract function name from tool_call block 0
Error parsing tool_call block 0: ...
```

### Monitoring Success Rate

```bash
# Count successful tool calls
grep "stop_reason.*tool_use" logs/proxy_*.log | wc -l

# Count failed tool calls (should be tool_use but got end_turn)
grep "stop_reason.*end_turn" logs/proxy_*.log | grep "tool_call" | wc -l

# Calculate success rate
# Success rate = tool_use / (tool_use + failed)
```

### Common Issues and Solutions

#### Issue 1: Tool calls not detected

**Symptoms:**
- Logs show `Found 0 tool_call blocks`
- Model output doesn't contain `<tool_call>` tags

**Possible causes:**
1. Model not properly configured for tool use
2. Tool definitions not sent correctly
3. Model choosing not to use tools

**Solution:**
- Check that tools are properly defined in the request
- Verify model is receiving tool definitions (check logs)
- Try with a simpler tool to test

#### Issue 2: Parsing errors

**Symptoms:**
- Logs show `Error parsing tool_call block`
- Tool calls detected but not converted

**Possible causes:**
1. Malformed ReAct format from model
2. Unexpected argument format
3. Special characters in arguments

**Solution:**
- Check the raw content in logs
- Verify argument values are properly formatted
- Report issue with example for further investigation

#### Issue 3: Complex arguments fail

**Symptoms:**
- Simple tools work, but complex tools (like TodoWrite) fail
- JSON parsing errors in logs

**Solution:**
- The enhanced parser (v1.0.1+) handles JSON arguments
- Ensure you're running the latest version
- Check logs for JSON parsing errors

## Performance Considerations

### Token Usage

GLM-4.7 ReAct format uses more tokens than standard function calling:

```
Standard: ~50 tokens for a tool call
ReAct: ~100-150 tokens (includes <think> tags and verbose format)
```

**Optimization:**
- The proxy removes `<think>` tags to reduce token usage in responses
- Consider this overhead when setting `MAX_TOKENS_LIMIT`

### Latency

**Non-streaming mode for tools:**
- Adds latency compared to streaming
- But ensures reliable tool call detection
- Trade-off is necessary for GLM-4.7 compatibility

**Typical latencies:**
- Simple tool call: 2-5 seconds
- Complex tool call (TodoWrite): 3-8 seconds
- Depends on model server performance

### Memory Usage

ReAct format parsing is memory-efficient:
- Regex-based parsing
- No additional model calls
- Minimal overhead

## Best Practices

### 1. Tool Design

**Keep tools simple:**
```python
# Good: Simple parameters
{
    "name": "get_weather",
    "parameters": {
        "location": {"type": "string"}
    }
}

# Avoid: Deeply nested structures
{
    "name": "complex_tool",
    "parameters": {
        "config": {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {...}
                }
            }
        }
    }
}
```

### 2. Error Handling

Always check `stop_reason` in responses:

```python
if response["stop_reason"] == "tool_use":
    # Process tool calls
    for block in response["content"]:
        if block["type"] == "tool_use":
            execute_tool(block["name"], block["input"])
else:
    # No tool calls, handle as text response
    print(response["content"][0]["text"])
```

### 3. Logging

Enable DEBUG logging during development:

```bash
# Development
LOG_LEVEL="DEBUG"

# Production
LOG_LEVEL="INFO"  # or "WARNING"
```

### 4. Testing

Test tool calls thoroughly:

```python
# test_tools.py
import httpx

def test_simple_tool():
    response = httpx.post(
        "http://localhost:8082/v1/messages",
        json={
            "model": "claude-opus-4-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }]
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["stop_reason"] == "tool_use"
    assert any(b["type"] == "tool_use" for b in data["content"])
```

## Comparison with Other Models

| Feature | GLM-4.7 | GPT-4o | DeepSeek V3 |
|---------|---------|--------|-------------|
| Function Calling Format | ReAct | Standard | Standard |
| Streaming Tool Calls | Limited | Full | Full |
| Tool Call Reliability | ~95% | ~99% | ~98% |
| Token Overhead | High | Low | Low |
| Setup Complexity | Medium | Low | Low |
| Cost | Low | High | Medium |

## Migration Path

If you need better tool call reliability, consider migrating:

### From GLM-4.7 to GPT-4o

```bash
# Change .env
OPENAI_API_KEY="sk-your-openai-key"
OPENAI_BASE_URL="https://api.openai.com/v1"
BIG_MODEL="gpt-4o"
MIDDLE_MODEL="gpt-4o"
SMALL_MODEL="gpt-4o-mini"
```

**Benefits:**
- Standard function calling (no ReAct parsing needed)
- Better streaming support
- Higher reliability (~99%)

**Trade-offs:**
- Higher cost
- May require VPN in some regions

### From GLM-4.7 to DeepSeek V3

```bash
# Change .env
OPENAI_API_KEY="your-deepseek-key"
OPENAI_BASE_URL="https://api.deepseek.com/v1"
BIG_MODEL="deepseek-chat"
MIDDLE_MODEL="deepseek-chat"
SMALL_MODEL="deepseek-chat"
```

**Benefits:**
- Standard function calling
- Good reliability (~98%)
- Lower cost than GPT-4o
- No regional restrictions

**Trade-offs:**
- Slightly lower quality than GPT-4o
- Smaller context window (64K vs 200K)

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for detailed debugging steps.

## Contributing

If you encounter issues with GLM-4.7 ReAct format parsing:

1. Enable DEBUG logging
2. Capture the problematic request/response
3. Create an issue with:
   - Raw model output (from logs)
   - Expected behavior
   - Actual behavior
   - Relevant log excerpts

## References

- [GLM-4 Documentation](https://github.com/THUDM/GLM-4)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Claude API Documentation](https://docs.anthropic.com/claude/reference/messages_post)
