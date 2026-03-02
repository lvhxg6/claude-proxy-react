"""Tests for compaction functionality.

This module tests the compaction boundary pruning, SSE conversion,
and usage statistics to ensure protocol compliance.
"""

from unittest.mock import Mock

import pytest

from src.conversion.request_converter import (
    _find_compaction_boundary,
    _prune_messages_at_boundary,
    convert_claude_to_openai,
)
from src.core.compaction import build_compaction_with_response
from src.models.claude import (
    ClaudeContentBlockCompaction,
    ClaudeContentBlockText,
    ClaudeMessage,
    ClaudeMessagesRequest,
)


class TestCompactionBoundaryPruning:
    """Test compaction boundary detection and pruning logic."""

    def test_find_compaction_boundary_with_valid_summary(self):
        """Test finding compaction boundary with non-empty summary."""
        messages = [
            ClaudeMessage(role="user", content="Hello"),
            ClaudeMessage(
                role="assistant",
                content=[
                    ClaudeContentBlockCompaction(type="compaction", content="This is a summary")
                ],
            ),
            ClaudeMessage(role="user", content="Continue"),
        ]

        boundary_index, summary = _find_compaction_boundary(messages)

        assert boundary_index == 1
        assert summary == "This is a summary"

    def test_find_compaction_boundary_with_null_summary(self):
        """Test finding compaction boundary with null/empty summary."""
        messages = [
            ClaudeMessage(role="user", content="Hello"),
            ClaudeMessage(
                role="assistant",
                content=[ClaudeContentBlockCompaction(type="compaction", content=None)],
            ),
            ClaudeMessage(role="user", content="Continue"),
        ]

        boundary_index, summary = _find_compaction_boundary(messages)

        assert boundary_index == 1
        assert summary is None

    def test_find_compaction_boundary_not_found(self):
        """Test when no compaction boundary exists."""
        messages = [
            ClaudeMessage(role="user", content="Hello"),
            ClaudeMessage(
                role="assistant",
                content=[ClaudeContentBlockText(type="text", text="Response")],
            ),
        ]

        boundary_index, summary = _find_compaction_boundary(messages)

        assert boundary_index == -1
        assert summary is None

    def test_prune_messages_preserves_from_boundary(self):
        """Test that pruning keeps messages from boundary onward."""
        messages = [
            ClaudeMessage(role="user", content="Message 1"),
            ClaudeMessage(role="assistant", content="Response 1"),
            ClaudeMessage(role="user", content="Message 2"),
            ClaudeMessage(
                role="assistant",
                content=[ClaudeContentBlockCompaction(type="compaction", content="Summary")],
            ),
            ClaudeMessage(role="user", content="Message 3"),
        ]

        pruned = _prune_messages_at_boundary(messages, 3, "Summary")

        assert len(pruned) == 2  # Messages at index 3 and 4
        assert pruned[0].role == "assistant"
        assert pruned[1].content == "Message 3"


class TestCompactionSafetyProtection:
    """Test safety protection: no pruning when summary is empty."""

    def test_convert_skips_pruning_when_summary_null(self):
        """Test that conversion skips pruning when summary is null."""
        request = ClaudeMessagesRequest(
            model="claude-opus-4",
            messages=[
                ClaudeMessage(role="user", content="Old message 1"),
                ClaudeMessage(role="assistant", content="Old response 1"),
                ClaudeMessage(
                    role="assistant",
                    content=[ClaudeContentBlockCompaction(type="compaction", content=None)],
                ),
                ClaudeMessage(role="user", content="New message"),
            ],
            max_tokens=1024,
        )

        mock_model_manager = Mock()
        mock_model_manager.map_claude_model_to_openai.return_value = "gpt-4"

        result = convert_claude_to_openai(request, mock_model_manager)

        # Should preserve all messages (no pruning)
        # System message (if any) + 4 original messages
        assert len(result["messages"]) == 4

    def test_convert_applies_pruning_when_summary_valid(self):
        """Test that conversion applies pruning when summary is valid."""
        request = ClaudeMessagesRequest(
            model="claude-opus-4",
            messages=[
                ClaudeMessage(role="user", content="Old message 1"),
                ClaudeMessage(role="assistant", content="Old response 1"),
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockCompaction(type="compaction", content="Valid summary")
                    ],
                ),
                ClaudeMessage(role="user", content="New message"),
            ],
            max_tokens=1024,
        )

        mock_model_manager = Mock()
        mock_model_manager.map_claude_model_to_openai.return_value = "gpt-4"

        result = convert_claude_to_openai(request, mock_model_manager)

        # Should only keep messages from boundary onward (2 messages)
        assert len(result["messages"]) == 2


class TestUsageStatistics:
    """Test usage token counting follows Anthropic protocol."""

    def test_usage_excludes_compaction_from_top_level(self):
        """Test that top-level output_tokens excludes compaction tokens."""
        response = build_compaction_with_response(
            summary="Test summary",
            content_blocks=[{"type": "text", "text": "Response"}],
            request=Mock(model="claude-opus-4"),
            stop_reason="end_turn",
            compaction_input_tokens=1000,
            compaction_output_tokens=200,
            message_input_tokens=500,
            message_output_tokens=100,
        )

        usage = response["usage"]

        # Top-level should only have message iteration tokens
        assert usage["input_tokens"] == 500
        assert usage["output_tokens"] == 100

        # Iterations should have both
        assert len(usage["iterations"]) == 2
        assert usage["iterations"][0]["type"] == "compaction"
        assert usage["iterations"][0]["output_tokens"] == 200
        assert usage["iterations"][1]["type"] == "message"
        assert usage["iterations"][1]["output_tokens"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
