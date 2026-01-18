# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Comprehensive debug logging for ReAct format parsing
- JSON parsing support for complex tool arguments (arrays/objects)
- Detailed tool call detection logging in response conversion

### Changed
- Enhanced `parse_glm_react_format` function with better error handling
- Made regex patterns case-insensitive for more robust parsing
- Improved streaming response ReAct format processing
- Replaced print statements with proper logger.debug calls

### Fixed
- Improved GLM-4.7 ReAct format tool call detection (addresses ~26% failure rate)
- Better handling of complex tool parameters like TodoWrite
- More robust parsing of tool calls with whitespace variations

## [1.0.0] - 2026-01-18

### Added
- Initial release of Claude Code Proxy
- Support for OpenAI-compatible API providers
- GLM-4.7 ReAct format parsing
- Streaming and non-streaming response support
- Custom headers support
- Model mapping (BIG_MODEL, MIDDLE_MODEL, SMALL_MODEL)
- Request cancellation support
- Comprehensive error handling and logging

### Features
- Full Claude API `/v1/messages` endpoint compatibility
- Function calling / tool use support
- Image input support (base64 encoded)
- Token counting endpoint
- Health check endpoints
- Azure OpenAI support
