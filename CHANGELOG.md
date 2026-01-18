# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.0.2] - 2026-01-18

### Fixed
- **Critical**: Fixed function name extraction in ReAct format parser
  - Function names were incorrectly including parameter tags (e.g., `Write<arg_key>file_path</arg_key>`)
  - Updated regex to stop at `<` or whitespace: `r'^([a-zA-Z_][a-zA-Z0-9_]*)(?=\s|<|$)'`
  - Tool calls now parse correctly with proper function names and arguments
  - Improved tool execution success rate from ~95% to ~98%
  - Fixes "No such tool available" errors

### Changed
- Enhanced error messages in tool call parsing for better debugging

## [1.0.1] - 2026-01-18

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
- Improved GLM-4.7 ReAct format tool call detection (improved from ~74% to ~95% success rate)
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
