"""Utilities for handling structured JSON annotations."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class JSONAnnotationHandler:
    """Handler for structured JSON annotations and metadata.

    Provides utilities for reading, writing, validating, and manipulating
    JSON-based annotations for summarization tasks.
    """

    def __init__(
        self,
        annotation_schema: Optional[Dict] = None,
        validate_schema: bool = True,
        auto_timestamp: bool = True,
    ):
        """Initialize JSON annotation handler.

        Args:
            annotation_schema: Expected schema for annotations (optional)
            validate_schema: Whether to validate annotations against schema
            auto_timestamp: Whether to automatically add timestamps
        """
        self.annotation_schema = annotation_schema
        self.validate_schema = validate_schema
        self.auto_timestamp = auto_timestamp

        # Default annotation schema for summarization tasks
        if self.annotation_schema is None:
            self.annotation_schema = self._get_default_schema()

    def _get_default_schema(self) -> Dict:
        """Get default annotation schema for summarization tasks.

        Returns:
            Default annotation schema
        """
        return {
            "type": "object",
            "required": ["id", "annotations"],
            "properties": {
                "id": {"type": "string"},
                "annotations": {
                    "type": "object",
                    "properties": {
                        "quality_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "factual_errors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "position": {"type": "integer"},
                                    "error_type": {"type": "string"},
                                    "severity": {
                                        "type": "string",
                                        "enum": ["low", "medium", "high"],
                                    },
                                    "correction": {"type": "string"},
                                },
                                "required": ["text", "error_type"],
                            },
                        },
                        "tool_calls": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tool": {
                                        "type": "string",
                                        "enum": ["search", "delete"],
                                    },
                                    "query": {"type": "string"},
                                    "position": {"type": "integer"},
                                    "results": {"type": "array"},
                                },
                                "required": ["tool", "query"],
                            },
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "annotator": {"type": "string"},
                                "timestamp": {"type": "string"},
                                "version": {"type": "string"},
                                "model_used": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }

    def load_annotations(self, file_path: Union[str, Path]) -> List[Dict]:
        """Load annotations from JSON file.

        Args:
            file_path: Path to JSON annotation file

        Returns:
            List of annotation records
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix == ".jsonl":
                    # Load JSONL format
                    annotations = []
                    for line_num, line in enumerate(f, 1):
                        try:
                            annotation = json.loads(line.strip())
                            annotations.append(annotation)
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Failed to parse line {line_num} in {file_path}: {e}"
                            )
                    return annotations
                else:
                    # Load regular JSON format
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    else:
                        return [data]

        except Exception as e:
            logger.error(f"Failed to load annotations from {file_path}: {e}")
            raise

    def save_annotations(
        self,
        annotations: List[Dict],
        file_path: Union[str, Path],
        format_type: str = "json",
    ):
        """Save annotations to JSON file.

        Args:
            annotations: List of annotation records
            file_path: Path to save annotations
            format_type: Output format ('json' or 'jsonl')
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                if format_type == "jsonl":
                    # Save as JSONL
                    for annotation in annotations:
                        json.dump(annotation, f, ensure_ascii=False)
                        f.write("\n")
                else:
                    # Save as regular JSON
                    json.dump(annotations, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(annotations)} annotations to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save annotations to {file_path}: {e}")
            raise

    def validate_annotation(self, annotation: Dict) -> Dict[str, Any]:
        """Validate annotation against schema.

        Args:
            annotation: Annotation to validate

        Returns:
            Validation result
        """
        result = {"is_valid": True, "errors": [], "warnings": []}

        if not self.validate_schema:
            return result

        try:
            # Basic structure validation
            if "id" not in annotation:
                result["errors"].append("Missing required field: id")

            if "annotations" not in annotation:
                result["errors"].append("Missing required field: annotations")
            else:
                # Validate annotation content
                annotation_content = annotation["annotations"]

                # Check quality score
                if "quality_score" in annotation_content:
                    score = annotation_content["quality_score"]
                    if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                        result["errors"].append(
                            "quality_score must be a number between 0 and 1"
                        )

                # Check factual errors
                if "factual_errors" in annotation_content:
                    errors = annotation_content["factual_errors"]
                    if not isinstance(errors, list):
                        result["errors"].append("factual_errors must be an array")
                    else:
                        for i, error in enumerate(errors):
                            if not isinstance(error, dict):
                                result["errors"].append(
                                    f"factual_errors[{i}] must be an object"
                                )
                                continue
                            if "text" not in error or "error_type" not in error:
                                result["errors"].append(
                                    f"factual_errors[{i}] missing required fields"
                                )

                # Check tool calls
                if "tool_calls" in annotation_content:
                    calls = annotation_content["tool_calls"]
                    if not isinstance(calls, list):
                        result["errors"].append("tool_calls must be an array")
                    else:
                        for i, call in enumerate(calls):
                            if not isinstance(call, dict):
                                result["errors"].append(
                                    f"tool_calls[{i}] must be an object"
                                )
                                continue
                            if "tool" not in call or "query" not in call:
                                result["errors"].append(
                                    f"tool_calls[{i}] missing required fields"
                                )
                            elif call["tool"] not in ["search", "delete"]:
                                result["errors"].append(
                                    f"tool_calls[{i}] invalid tool type: {call['tool']}"
                                )

        except Exception as e:
            result["errors"].append(f"Validation error: {str(e)}")

        if result["errors"]:
            result["is_valid"] = False

        return result

    def create_annotation(
        self,
        sample_id: str,
        quality_score: Optional[float] = None,
        factual_errors: Optional[List[Dict]] = None,
        tool_calls: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Create a new annotation record.

        Args:
            sample_id: ID of the sample being annotated
            quality_score: Quality score (0-1)
            factual_errors: List of factual error annotations
            tool_calls: List of tool call annotations
            metadata: Additional metadata

        Returns:
            New annotation record
        """
        annotation = {"id": sample_id, "annotations": {}}

        if quality_score is not None:
            annotation["annotations"]["quality_score"] = quality_score

        if factual_errors is not None:
            annotation["annotations"]["factual_errors"] = factual_errors

        if tool_calls is not None:
            annotation["annotations"]["tool_calls"] = tool_calls

        # Add metadata
        annotation_metadata = metadata or {}
        if self.auto_timestamp:
            annotation_metadata["timestamp"] = datetime.now().isoformat()

        if annotation_metadata:
            annotation["annotations"]["metadata"] = annotation_metadata

        return annotation

    def merge_annotations(self, base_annotation: Dict, update_annotation: Dict) -> Dict:
        """Merge two annotation records.

        Args:
            base_annotation: Base annotation to update
            update_annotation: Annotation updates to apply

        Returns:
            Merged annotation record
        """
        merged = base_annotation.copy()

        if "annotations" in update_annotation:
            if "annotations" not in merged:
                merged["annotations"] = {}

            merged_annotations = merged["annotations"]
            update_annotations = update_annotation["annotations"]

            # Merge fields
            for key, value in update_annotations.items():
                if (
                    key in ["factual_errors", "tool_calls"]
                    and key in merged_annotations
                ):
                    # Merge lists
                    if isinstance(value, list) and isinstance(
                        merged_annotations[key], list
                    ):
                        merged_annotations[key].extend(value)
                    else:
                        merged_annotations[key] = value
                elif key == "metadata" and key in merged_annotations:
                    # Merge metadata objects
                    if isinstance(value, dict) and isinstance(
                        merged_annotations[key], dict
                    ):
                        merged_annotations[key].update(value)
                    else:
                        merged_annotations[key] = value
                else:
                    # Replace value
                    merged_annotations[key] = value

        return merged

    def extract_tool_calls(self, annotation: Dict) -> List[Dict]:
        """Extract tool calls from annotation.

        Args:
            annotation: Annotation record

        Returns:
            List of tool call records
        """
        if "annotations" not in annotation:
            return []

        tool_calls = annotation["annotations"].get("tool_calls", [])
        return tool_calls if isinstance(tool_calls, list) else []

    def extract_factual_errors(self, annotation: Dict) -> List[Dict]:
        """Extract factual errors from annotation.

        Args:
            annotation: Annotation record

        Returns:
            List of factual error records
        """
        if "annotations" not in annotation:
            return []

        factual_errors = annotation["annotations"].get("factual_errors", [])
        return factual_errors if isinstance(factual_errors, list) else []

    def filter_annotations(
        self,
        annotations: List[Dict],
        quality_threshold: Optional[float] = None,
        error_types: Optional[List[str]] = None,
        tool_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Filter annotations based on criteria.

        Args:
            annotations: List of annotations to filter
            quality_threshold: Minimum quality score
            error_types: List of error types to include
            tool_types: List of tool types to include

        Returns:
            Filtered list of annotations
        """
        filtered = []

        for annotation in annotations:
            include = True

            if "annotations" not in annotation:
                continue

            annotation_content = annotation["annotations"]

            # Check quality threshold
            if quality_threshold is not None:
                quality_score = annotation_content.get("quality_score")
                if quality_score is None or quality_score < quality_threshold:
                    include = False

            # Check error types
            if error_types is not None and include:
                factual_errors = annotation_content.get("factual_errors", [])
                if not any(
                    error.get("error_type") in error_types for error in factual_errors
                ):
                    include = False

            # Check tool types
            if tool_types is not None and include:
                tool_calls = annotation_content.get("tool_calls", [])
                if not any(call.get("tool") in tool_types for call in tool_calls):
                    include = False

            if include:
                filtered.append(annotation)

        return filtered

    def get_annotation_stats(self, annotations: List[Dict]) -> Dict[str, Any]:
        """Get statistics about annotations.

        Args:
            annotations: List of annotations to analyze

        Returns:
            Annotation statistics
        """
        stats = {
            "total_annotations": len(annotations),
            "quality_scores": [],
            "error_types": {},
            "tool_types": {},
            "has_quality_score": 0,
            "has_factual_errors": 0,
            "has_tool_calls": 0,
        }

        for annotation in annotations:
            if "annotations" not in annotation:
                continue

            annotation_content = annotation["annotations"]

            # Quality scores
            if "quality_score" in annotation_content:
                stats["has_quality_score"] += 1
                stats["quality_scores"].append(annotation_content["quality_score"])

            # Factual errors
            if "factual_errors" in annotation_content:
                errors = annotation_content["factual_errors"]
                if errors:
                    stats["has_factual_errors"] += 1
                    for error in errors:
                        error_type = error.get("error_type", "unknown")
                        stats["error_types"][error_type] = (
                            stats["error_types"].get(error_type, 0) + 1
                        )

            # Tool calls
            if "tool_calls" in annotation_content:
                calls = annotation_content["tool_calls"]
                if calls:
                    stats["has_tool_calls"] += 1
                    for call in calls:
                        tool_type = call.get("tool", "unknown")
                        stats["tool_types"][tool_type] = (
                            stats["tool_types"].get(tool_type, 0) + 1
                        )

        # Calculate averages
        if stats["quality_scores"]:
            stats["avg_quality_score"] = sum(stats["quality_scores"]) / len(
                stats["quality_scores"]
            )
            stats["min_quality_score"] = min(stats["quality_scores"])
            stats["max_quality_score"] = max(stats["quality_scores"])
        else:
            stats["avg_quality_score"] = None
            stats["min_quality_score"] = None
            stats["max_quality_score"] = None

        return stats
