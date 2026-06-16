"""Qwen-based object extractor for discovery layer."""

import logging
from pathlib import Path

from .base import ObjectDiscoverer

logger = logging.getLogger(__name__)


class QwenExtractor(ObjectDiscoverer):
    """
    Extract objects from task instructions using Qwen LLM.

    Uses a small local Qwen model for offline extraction.
    """

    def __init__(self, model_path: Path | None = None):
        """
        Initialize Qwen extractor.

        Args:
            model_path: Path to Qwen model checkpoint
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

        if model_path and model_path.exists():
            self._load_model()
        else:
            logger.warning(
                f"Qwen model path not set or doesn't exist: {model_path}. "
                "Will use mock extraction."
            )

    def _load_model(self) -> None:
        """
        Load Qwen model and tokenizer.

        TODO: Implement actual model loading after Qwen deployment.
        """
        # TODO: verify API after Qwen deployment
        # Expected API (HuggingFace Transformers style):
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        logger.info(f"Loading Qwen model from {self.model_path}")
        logger.warning("Qwen model loading not implemented yet - using mock extraction")

    def discover_objects(self, instruction: str, config: dict) -> list[str]:
        """
        Extract objects from instruction using Qwen.

        Args:
            instruction: Task instruction string
            config: Discovery config dict (uses always_include)

        Returns:
            Deduplicated list of object query strings
        """
        if self.model is None:
            # Mock extraction for testing before model deployment
            return self._mock_extract(instruction, config)

        # TODO: verify API after Qwen deployment
        # Construct prompt
        prompt = self._build_prompt(instruction)

        # Generate
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model.generate(**inputs, max_new_tokens=100)
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse response
        # objects = self._parse_response(response)

        # Add always_include
        always_include = config.get("always_include", [])
        # objects.extend(always_include)

        # Normalize
        # return self._normalize_queries(objects)

        return self._mock_extract(instruction, config)

    def _build_prompt(self, instruction: str) -> str:
        """
        Build prompt for Qwen model.

        Args:
            instruction: Task instruction

        Returns:
            Formatted prompt string
        """
        prompt = f"""Given a robot manipulation task instruction, extract all objects that need to be detected/segmented.
Return a comma-separated list of object names.

Instruction: {instruction}

Objects:"""
        return prompt

    def _parse_response(self, response: str) -> list[str]:
        """
        Parse Qwen response to extract object list.

        Args:
            response: Raw model output

        Returns:
            List of object strings
        """
        # Extract after "Objects:" marker
        if "Objects:" in response:
            response = response.split("Objects:")[-1]

        # Split by comma
        objects = [obj.strip() for obj in response.split(",")]
        return [obj for obj in objects if obj]

    def _mock_extract(self, instruction: str, config: dict) -> list[str]:
        """
        Mock extraction for testing before model deployment.

        Uses simple heuristics similar to rule-based extractor.

        Args:
            instruction: Task instruction
            config: Discovery config

        Returns:
            List of object queries
        """
        logger.debug(f"Using mock extraction for: {instruction}")

        # Simple mock: extract words that look like objects
        instruction_lower = instruction.lower()
        mock_objects = set()

        # Look for common object keywords
        object_keywords = [
            "cup", "block", "drawer", "spoon", "fork", "knife",
            "plate", "bowl", "bottle", "can", "box", "toy",
            "red", "blue", "green", "yellow", "black", "white",
        ]

        words = instruction_lower.split()
        for i, word in enumerate(words):
            word_clean = word.strip(".,!?")
            if word_clean in object_keywords:
                # Check if preceded by color
                if i > 0 and words[i - 1].strip(".,!?") in ["red", "blue", "green", "yellow", "black", "white"]:
                    mock_objects.add(f"{words[i - 1].strip('.,!?')} {word_clean}")
                else:
                    mock_objects.add(word_clean)

        # Add always_include
        always_include = config.get("always_include", [])
        mock_objects.update(item.lower() for item in always_include)

        result = sorted(mock_objects)
        logger.info(f"QwenExtractor (mock): extracted {len(result)} objects")

        return result

    def _normalize_queries(self, queries: list[str]) -> list[str]:
        """
        Normalize and deduplicate queries.

        Args:
            queries: List of raw queries

        Returns:
            Normalized list
        """
        normalized = set()
        for q in queries:
            q = q.strip().lower()
            if q:
                normalized.add(q)
        return sorted(normalized)
