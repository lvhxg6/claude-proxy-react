#!/usr/bin/env python3
"""Test script to verify GLM-4.7 ReAct format conversion."""

import asyncio
import json
import httpx


async def test_react_format_conversion():
    """Test that the proxy correctly converts GLM-4.7 ReAct format to Claude format."""

    # Proxy endpoint
    proxy_url = "http://localhost:8082/v1/messages"

    # Test request with tool definition
    request_data = {
        "model": "claude-opus-4-5",  # Will be mapped to GLM-4.7-FP8
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like in San Francisco today? Use the get_weather tool."
            }
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
    }

    print("=" * 80)
    print("Testing ReAct Format Conversion")
    print("=" * 80)
    print(f"\nSending request to proxy: {proxy_url}")
    print(f"Model: {request_data['model']}")
    print(f"Tools: {[tool['name'] for tool in request_data['tools']]}")
    print("\n" + "-" * 80)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                proxy_url,
                json=request_data,
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
            )

            print(f"\nResponse Status: {response.status_code}")
            print("-" * 80)

            if response.status_code == 200:
                response_data = response.json()
                print("\n✅ SUCCESS: Received valid response")
                print("\nResponse structure:")
                print(f"  - ID: {response_data.get('id')}")
                print(f"  - Type: {response_data.get('type')}")
                print(f"  - Role: {response_data.get('role')}")
                print(f"  - Model: {response_data.get('model')}")
                print(f"  - Stop reason: {response_data.get('stop_reason')}")

                content_blocks = response_data.get('content', [])
                print(f"\nContent blocks: {len(content_blocks)}")

                for i, block in enumerate(content_blocks):
                    print(f"\n  Block {i}:")
                    print(f"    Type: {block.get('type')}")

                    if block.get('type') == 'text':
                        text = block.get('text', '')
                        print(f"    Text: {text[:100]}{'...' if len(text) > 100 else ''}")

                    elif block.get('type') == 'tool_use':
                        print(f"    Tool ID: {block.get('id')}")
                        print(f"    Tool Name: {block.get('name')}")
                        print(f"    Tool Input: {json.dumps(block.get('input', {}), indent=6)}")

                # Check if tool calls were detected
                has_tool_use = any(block.get('type') == 'tool_use' for block in content_blocks)

                if has_tool_use:
                    print("\n✅ Tool calls detected and converted successfully!")
                else:
                    print("\n⚠️  No tool calls detected in response")
                    print("    This might be expected if the model didn't use tools")

                print("\n" + "=" * 80)
                print("Full Response:")
                print("=" * 80)
                print(json.dumps(response_data, indent=2, ensure_ascii=False))

            else:
                print(f"\n❌ ERROR: Unexpected status code {response.status_code}")
                print("\nResponse body:")
                print(response.text)

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        print("\nTraceback:")
        print(traceback.format_exc())


async def test_streaming_react_format():
    """Test streaming response with ReAct format."""

    proxy_url = "http://localhost:8082/v1/messages"

    request_data = {
        "model": "claude-opus-4-5",
        "max_tokens": 1024,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Tokyo? Use the get_weather tool."
            }
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state or country"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
    }

    print("\n" + "=" * 80)
    print("Testing Streaming ReAct Format Conversion")
    print("=" * 80)
    print(f"\nSending streaming request to proxy: {proxy_url}")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                proxy_url,
                json=request_data,
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
            ) as response:
                print(f"\nResponse Status: {response.status_code}")
                print("-" * 80)
                print("\nStreaming events:\n")

                event_count = 0
                tool_use_detected = False

                async for line in response.aiter_lines():
                    if line.strip():
                        print(line)

                        # Check for tool_use events
                        if 'tool_use' in line:
                            tool_use_detected = True

                        event_count += 1

                print(f"\n\nTotal events received: {event_count}")

                if tool_use_detected:
                    print("✅ Tool use events detected in streaming response!")
                else:
                    print("⚠️  No tool use events detected")

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        print("\nTraceback:")
        print(traceback.format_exc())


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("GLM-4.7 ReAct Format Conversion Test Suite")
    print("=" * 80)

    # Test non-streaming
    await test_react_format_conversion()

    # Wait a bit between tests
    await asyncio.sleep(2)

    # Test streaming
    await test_streaming_react_format()

    print("\n" + "=" * 80)
    print("Test Suite Complete")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
