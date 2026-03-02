"""GLM-4.7 tokenizer management with lazy loading and caching."""

import logging
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

# Local tokenizer file path (relative to project root)
_LOCAL_TOKENIZER_PATH = Path(__file__).parent.parent.parent / "tokenizer.json"
_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "huggingface"


class GLMTokenizer:
    """Lazy-loading GLM-4.7 tokenizer singleton."""

    _instance: Optional[Tokenizer] = None
    _repo_id = "zai-org/GLM-4.7"
    _mirror_endpoint = "https://hf-mirror.com"

    @classmethod
    def get_tokenizer(cls) -> Tokenizer:
        """Get or initialize the GLM-4.7 tokenizer.

        Loading order:
        1. Local tokenizer.json in project root
        2. HuggingFace cache (.cache/huggingface/)
        3. Download from HuggingFace (with mirror fallback)
        """
        if cls._instance is not None:
            return cls._instance

        # 1. Try local file first
        if _LOCAL_TOKENIZER_PATH.exists():
            try:
                logger.info(f"Loading GLM-4.7 tokenizer from local file: {_LOCAL_TOKENIZER_PATH}")
                cls._instance = Tokenizer.from_file(str(_LOCAL_TOKENIZER_PATH))
                logger.info("GLM-4.7 tokenizer loaded from local file")
                return cls._instance
            except Exception as e:
                logger.warning(f"Failed to load local tokenizer: {e}")

        # 2. Try downloading (official endpoint, then mirror)
        endpoints = [None, cls._mirror_endpoint]  # None = default endpoint
        env_endpoint = os.environ.get("HF_ENDPOINT")
        if env_endpoint:
            endpoints.insert(0, env_endpoint)

        last_error = None
        for endpoint in endpoints:
            try:
                label = endpoint or "huggingface.co"
                logger.info(f"Downloading GLM-4.7 tokenizer from {label}")
                kwargs = {
                    "repo_id": cls._repo_id,
                    "filename": "tokenizer.json",
                    "cache_dir": str(_CACHE_DIR),
                }
                if endpoint:
                    kwargs["endpoint"] = endpoint
                tokenizer_path = hf_hub_download(**kwargs)
                cls._instance = Tokenizer.from_file(tokenizer_path)
                logger.info(f"GLM-4.7 tokenizer loaded from {label}")
                return cls._instance
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to download from {label}: {e}")

        raise RuntimeError(
            f"Cannot initialize GLM-4.7 tokenizer. "
            f"Place tokenizer.json in project root or fix network. Last error: {last_error}"
        )

    @classmethod
    def encode(cls, text: str) -> list[int]:
        """Encode text to token IDs."""
        tokenizer = cls.get_tokenizer()
        return tokenizer.encode(text).ids

    @classmethod
    def count_tokens(cls, text: str) -> int:
        """Count tokens in text."""
        return len(cls.encode(text))
