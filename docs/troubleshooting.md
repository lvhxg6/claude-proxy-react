# Troubleshooting Guide

## Tool Call Issues with GLM-4.7

### Problem: Tool calls fail with "No such tool available" error

**Symptoms:**
- Tool calls fail with error: `Error: No such tool available: Write<arg_key>file_path</arg_key>...`
- Function name includes parameter tags instead of just the function name
- Arguments are empty: `'arguments': '{}'`
- `stop_reason` may be `tool_use` but tool execution fails

**Root Cause:**
GLM-4.7-FP8 returns ReAct format where function name and parameters are concatenated without separators:
```
<tool_call>Write<arg_key>file_path</arg_key><arg_value>...</arg_value></tool_call>
```

The original parser used regex `^([a-zA-Z_][a-zA-Z0-9_]*)` which would match beyond the function name, including the `<arg_key>` tag.

**Solution (Fixed in v1.0.2):**
Updated the function name extraction regex to stop at `<` or whitespace:
```python
name_match = re.search(r'^([a-zA-Z_][a-zA-Z0-9_]*)(?=\s|<|$)', match.strip())
```

This ensures:
- Function name: `Write` (correct)
- Arguments: `{"file_path": "...", "content": "..."}` (correct)

### Problem: TodoWrite or other tools not being executed

**Symptoms:**
- Claude Code CLI shows tool definitions being sent
- Model responds with text but tools are not executed
- `stop_reason` is `end_turn` instead of `tool_use`

**Root Cause:**
GLM-4.7-FP8 uses a non-standard ReAct format for tool calls instead of OpenAI's standard function calling format.

**Solution:**
The proxy has been enhanced (v1.0.2+) with improved ReAct format parsing that correctly handles concatenated function names and parameters.

1. **Enable DEBUG logging** in `.env`:
   ```bash
   LOG_LEVEL="DEBUG"
   ```

2. **Restart the proxy**:
   ```bash
   python start_proxy.py
   ```

3. **Check logs** for ReAct parsing:
   ```bash
   tail -f logs/proxy_*.log | grep "ReAct"
   ```

4. **Look for these log entries**:
   - `=== Parsing ReAct Format ===` - Parser is being invoked
   - `Found X tool_call blocks` - Tool calls detected
   - `Successfully parsed tool call: ToolName` - Successful parsing
   - `Parsed X tool calls from ReAct format` - Final count

**Expected behavior after fix:**
- Tool calls should be detected and parsed correctly
- Function names extracted without parameter tags
- Arguments properly parsed from `<arg_key>/<arg_value>` pairs
- `stop_reason` should be `tool_use` when tools are called
- Tool execution success rate should be >98% (previously ~74%, then ~95%)

### Debugging Tool Call Failures

If tool calls still fail after the fix:

1. **Check the raw response** in logs:
   ```bash
   grep "OpenAI Response:" logs/proxy_*.log | tail -1
   ```

2. **Verify ReAct format** - Look for:
   ```
   <tool_call>ToolName<arg_key>param</arg_key><arg_value>value</arg_value></tool_call>
   ```

3. **Check for parsing errors**:
   ```bash
   grep "Error parsing tool_call block" logs/proxy_*.log
   ```

4. **Common issues**:
   - **Function name includes tags** (Fixed in v1.0.2): Function name now stops at `<` character
   - **Malformed ReAct tags**: Check for typos or missing closing tags
   - **Complex JSON in arguments**: Should now be handled correctly
   - **Case sensitivity**: Now handled with `re.IGNORECASE`
   - **Extra whitespace**: Now handled with `\s*` in regex

### Alternative: Use a More Compatible Model

If GLM-4.7 compatibility issues persist, consider switching to a model with full OpenAI function calling support:

**Recommended alternatives:**

1. **OpenAI GPT-4o** (Best compatibility):
   ```bash
   OPENAI_API_KEY="sk-your-key"
   OPENAI_BASE_URL="https://api.openai.com/v1"
   BIG_MODEL="gpt-4o"
   MIDDLE_MODEL="gpt-4o"
   SMALL_MODEL="gpt-4o-mini"
   ```

2. **DeepSeek V3** (Chinese alternative):
   ```bash
   OPENAI_API_KEY="your-deepseek-key"
   OPENAI_BASE_URL="https://api.deepseek.com/v1"
   BIG_MODEL="deepseek-chat"
   MIDDLE_MODEL="deepseek-chat"
   SMALL_MODEL="deepseek-chat"
   ```

3. **Qwen-Plus** (Alibaba Cloud):
   ```bash
   OPENAI_API_KEY="your-qwen-key"
   OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
   BIG_MODEL="qwen-plus"
   MIDDLE_MODEL="qwen-plus"
   SMALL_MODEL="qwen-turbo"
   ```

## Connection Issues

### Proxy won't start

**Error: `OPENAI_API_KEY not found`**

Solution:
```bash
# Create .env file
cp .env.example .env
# Edit and add your API key
nano .env
```

**Error: `Port 8082 already in use`**

Solution:
```bash
# Change port in .env
PORT="8083"
```

### Claude Code CLI can't connect

**Error: `Connection refused`**

Check:
1. Proxy is running: `ps aux | grep claude-code-proxy`
2. Port is correct: `netstat -an | grep 8082`
3. Firewall allows connections

**Error: `Invalid API key`**

If you set `ANTHROPIC_API_KEY` in the proxy's `.env`, you must use the exact same key when running Claude Code CLI:

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_API_KEY="exact-matching-key" claude
```

## Performance Issues

### Slow responses

1. **Check model configuration**:
   - GLM-4.7-FP8 is quantized and should be fast
   - If using remote API, check network latency

2. **Enable connection pooling** (already enabled by default)

3. **Check timeout settings**:
   ```bash
   REQUEST_TIMEOUT="120"  # Increase if needed
   ```

### High memory usage

1. **Reduce max tokens**:
   ```bash
   MAX_TOKENS_LIMIT="4096"  # Lower if needed
   ```

2. **Check for memory leaks**:
   ```bash
   # Monitor memory usage
   watch -n 1 'ps aux | grep claude-code-proxy'
   ```

## Streaming Issues

### Streaming responses incomplete

**Symptoms:**
- Responses cut off mid-sentence
- Tool calls not completed

**Solution:**
1. Check timeout settings
2. Verify network stability
3. Check logs for errors:
   ```bash
   grep "Streaming error" logs/proxy_*.log
   ```

### Tool calls in streaming mode

**Note:** When tools are present, the proxy automatically uses non-streaming backend but returns SSE format to the client. This is by design to work around GLM-4.7 limitations.

Check logs for:
```
Tools present: using non-streaming backend but returning SSE format to client
```

## Logging and Debugging

### Enable detailed logging

```bash
# In .env
LOG_LEVEL="DEBUG"
```

### Log file location

```
logs/proxy_YYYYMMDD.log
```

### Useful log searches

```bash
# Tool call detection
grep "Tool Call Detection" logs/proxy_*.log

# ReAct parsing
grep "Parsing ReAct Format" logs/proxy_*.log

# Errors
grep "ERROR" logs/proxy_*.log

# Tool call success rate
grep "stop_reason.*tool_use" logs/proxy_*.log | wc -l
grep "stop_reason.*end_turn" logs/proxy_*.log | wc -l
```

## Getting Help

If you encounter issues not covered here:

1. **Check logs** with DEBUG level enabled
2. **Search existing issues**: https://github.com/anthropics/claude-code/issues
3. **Create a new issue** with:
   - Proxy version
   - Model being used (GLM-4.7, GPT-4o, etc.)
   - Relevant log excerpts
   - Steps to reproduce

## Known Limitations

### GLM-4.7-FP8 Specific

1. **ReAct format only**: Does not support standard OpenAI function calling
2. **Streaming tool calls**: May have issues with complex tool parameters
3. **Tool call reliability**: ~98% success rate (improved from ~74% → ~95% → ~98%)

### General

1. **Image support**: Limited to base64 encoded images
2. **Model mapping**: Fixed mapping (haiku→SMALL, sonnet→MIDDLE, opus→BIG)
3. **Token counting**: Uses character-based estimation (4 chars = 1 token)
