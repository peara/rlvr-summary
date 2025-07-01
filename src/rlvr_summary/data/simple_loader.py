"""Simple data loader utilities for quick testing and prototyping."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


class SimpleDataLoader:
    """Simple data loader for basic use cases and testing.

    Provides a minimal interface for loading data from various formats
    without the full complexity of the specialized loaders.
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        format: str = "json",
    ):
        """Initialize simple data loader.

        Args:
            data_path: Path to data file or directory
            format: Data format ('json', 'jsonl', 'txt')
        """
        self.data_path = Path(data_path) if data_path else None
        self.format = format.lower()

    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load data from the specified path.

        Args:
            limit: Maximum number of items to load

        Returns:
            List of data items
        """
        if not self.data_path or not self.data_path.exists():
            logger.warning(f"Data path not found: {self.data_path}")
            return []

        if self.format == "json":
            return self._load_json(limit)
        elif self.format == "jsonl":
            return self._load_jsonl(limit)
        elif self.format == "txt":
            return self._load_txt(limit)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def _load_json(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load JSON data."""
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return data[:limit] if limit else data
            else:
                return [data]
        except Exception as e:
            logger.error(f"Error loading JSON file {self.data_path}: {e}")
            return []

    def _load_jsonl(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load JSONL data."""
        data = []
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    if limit and len(data) >= limit:
                        break

                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num + 1}: {e}")

        except Exception as e:
            logger.error(f"Error loading JSONL file {self.data_path}: {e}")

        return data

    def _load_txt(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load text data."""
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if limit:
                lines = lines[:limit]

            return [{"text": line.strip()} for line in lines if line.strip()]

        except Exception as e:
            logger.error(f"Error loading text file {self.data_path}: {e}")
            return []


def create_data_loader(
    data_path: Union[str, Path], format: Optional[str] = None, **kwargs
) -> SimpleDataLoader:
    """Create a simple data loader with automatic format detection.

    Args:
        data_path: Path to data file
        format: Data format (auto-detected if None)
        **kwargs: Additional arguments for SimpleDataLoader

    Returns:
        Configured SimpleDataLoader instance
    """
    data_path = Path(data_path)

    if format is None:
        # Auto-detect format from file extension
        suffix = data_path.suffix.lower()
        if suffix == ".json":
            format = "json"
        elif suffix == ".jsonl":
            format = "jsonl"
        elif suffix in [".txt", ".text"]:
            format = "txt"
        else:
            logger.warning(f"Unknown file extension {suffix}, defaulting to json")
            format = "json"

    return SimpleDataLoader(data_path=data_path, format=format, **kwargs)
