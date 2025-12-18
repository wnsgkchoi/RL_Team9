"""
Reward Evaluator for grid-based navigation environments.

This module provides a class for evaluating reward values for each cell in a grid map
using vision language models (specifically ChatGPT-5).
"""

from __future__ import annotations

import os
import re
import json
from typing import List, Optional, Tuple

from model.mllm.mllms.chatgpt5.client import ChatGPT5_Client
from model.mllm.data_structure.input import GenerationInput, GenerationParameters, APIParameters
from reward_evaluator.prompts import REWARD_EVALUATION_SYSTEM_PROMPT, REWARD_EVALUATION_PROMPT








def main():
    
    # Example usage
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use a default test image if available
        test_image = "/root/omdr_workspace/src/reward_evaluator/map_4.png"
        if os.path.exists(test_image):
            image_path = test_image
        else:
            print("Usage: python reward_evaluator.py <path_to_map_image>")
            sys.exit(1)

    # Create evaluator and run
    evaluator = RewardEvaluator(reasoning_effort = "medium")

    try:
        rewards = evaluator.run(image_path)

        print("\n" + "="*60)
        print("REWARD MATRIX")
        print("="*60)
        for i, row in enumerate(rewards):
            print(f"Row {i:2d}: {' '.join(f'{val:6.3f}' for val in row)}")
        print("="*60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()






class RewardEvaluator:
    """
    Evaluates reward values for each cell in a grid map using ChatGPT-5.

    The evaluator uses vision-language models to analyze a map image and assign
    reward values to each cell based on the distance from the goal, with the highest
    rewards at the goal and decreasing rewards as distance increases.

    Example usage:
        evaluator = RewardEvaluator()
        rewards = evaluator.run("/path/to/map.png")
        # rewards is a 2D list of reward values
    """

    def __init__(
        self,
        reasoning_effort: str = "medium",
        discount_factor: float = 0.9
    ) -> None:
        """
        Initialize the RewardEvaluator.

        Args:
            reasoning_effort: The reasoning effort level for ChatGPT-5
                            ("low", "medium", or "high")
            discount_factor: Discount factor for reward propagation. Rewards decrease
                           as discount_factor^distance from the goal.
        """
        self.model = ChatGPT5_Client()
        self.model.initiate()
        self.reasoning_effort = reasoning_effort
        self.discount_factor = discount_factor
        
        return



    def run(
        self,
        image_path: str,
        grid_size: Optional[Tuple[int, int]] = None
    ) -> List[List[float]]:
        """
        Evaluate reward values for each cell in the grid map.

        Args:
            image_path: Path to the map image file
            grid_size: Optional tuple of (rows, cols) if known. If not provided,
                      will be extracted from the model's response.

        Returns:
            A 2D list of reward values with the same dimensions as the map grid.
            Each element represents the reward for the corresponding cell.

        Raises:
            FileNotFoundError: If the image file does not exist
            ValueError: If the model's response cannot be parsed
        """
        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Create generation input with image and prompt
        gen_input = GenerationInput(
            text_prompt=REWARD_EVALUATION_PROMPT(self.discount_factor),
            system_prompt=REWARD_EVALUATION_SYSTEM_PROMPT,
            image_paths=[image_path]
        )

        # Set generation parameters
        parameters = GenerationParameters(
            api_parameters=APIParameters(effort=self.reasoning_effort)
        )

        # Call the model
        print(f"[RewardEvaluator] Analyzing map: {image_path}")
        print(f"[RewardEvaluator] Using reasoning effort: {self.reasoning_effort}")

        output, metadata = self.model.infer(gen_input, parameters)

        # Print metadata for debugging
        if metadata.api_usage:
            print(f"[RewardEvaluator] API Usage:")
            print(f"  Input tokens: {metadata.api_usage.input_tokens}")
            print(f"  Output tokens: {metadata.api_usage.output_tokens}")
            print(f"  Estimated cost: ${metadata.api_usage.estimated_cost_usd:.4f}")

        # Parse the response
        rewards = self._parse_response(output.text, grid_size)

        print(f"[RewardEvaluator] Successfully extracted reward matrix of size {len(rewards)}x{len(rewards[0])}")

        return rewards


    def run_text(
        self,
        text_map: str,
        grid_size: Optional[Tuple[int, int]] = None
    ) -> List[List[float]]:
        """
        Evaluate reward values for each cell in the grid map.

        Args:
            image_path: Path to the map image file
            grid_size: Optional tuple of (rows, cols) if known. If not provided,
                      will be extracted from the model's response.

        Returns:
            A 2D list of reward values with the same dimensions as the map grid.
            Each element represents the reward for the corresponding cell.

        Raises:
            FileNotFoundError: If the image file does not exist
            ValueError: If the model's response cannot be parsed
        """
        # Create generation input with image and prompt
        gen_input = GenerationInput(
            text_prompt=REWARD_EVALUATION_PROMPT(self.discount_factor),
            system_prompt=REWARD_EVALUATION_SYSTEM_PROMPT,
            text_map=text_map
        )

        # Set generation parameters
        parameters = GenerationParameters(
            api_parameters=APIParameters(effort=self.reasoning_effort)
        )

        # Call the model
        print(f"[RewardEvaluator] Using reasoning effort: {self.reasoning_effort}")

        output, metadata = self.model.infer(gen_input, parameters)

        # Print metadata for debugging
        if metadata.api_usage:
            print(f"[RewardEvaluator] API Usage:")
            print(f"  Input tokens: {metadata.api_usage.input_tokens}")
            print(f"  Output tokens: {metadata.api_usage.output_tokens}")
            print(f"  Estimated cost: ${metadata.api_usage.estimated_cost_usd:.4f}")

        # Parse the response
        rewards = self._parse_response(output.text, grid_size)

        print(f"[RewardEvaluator] Successfully extracted reward matrix of size {len(rewards)}x{len(rewards[0])}")

        return rewards


    def _parse_response(
        self,
        response_text: str,
        expected_grid_size: Optional[Tuple[int, int]] = None
    ) -> List[List[float]]:
        """
        Parse the model's response to extract the reward matrix.

        Args:
            response_text: The text output from the model (expected to be JSON)
            expected_grid_size: Optional expected (rows, cols) for validation

        Returns:
            A 2D list of reward values

        Raises:
            ValueError: If the response cannot be parsed or has unexpected dimensions
        """
        # Try to parse as JSON
        try:
            # First, try to parse the response directly as JSON
            data = json.loads(response_text)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    raise ValueError("Found code block but could not parse as JSON")
            else:
                # Try to find any JSON object in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        raise ValueError("Found JSON-like structure but could not parse")
                else:
                    raise ValueError("Could not find valid JSON in model response")

        # Validate JSON structure
        if not isinstance(data, dict):
            raise ValueError("Response is not a JSON object")

        if "grid_size" not in data or "rewards" not in data:
            raise ValueError("JSON missing required fields: 'grid_size' and/or 'rewards'")

        # Extract grid size
        grid_size = data["grid_size"]
        if not isinstance(grid_size, dict) or "rows" not in grid_size or "cols" not in grid_size:
            raise ValueError("Invalid 'grid_size' format in JSON")

        rows = int(grid_size["rows"])
        cols = int(grid_size["cols"])
        print(f"[RewardEvaluator] Detected grid size: {rows}x{cols}")

        # Extract rewards
        rewards = data["rewards"]
        if not isinstance(rewards, list):
            raise ValueError("'rewards' field is not a list")

        # Convert all values to float and validate structure
        try:
            rewards = [[float(val) for val in row] for row in rewards]
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid reward values in JSON: {e}")

        # Validate the result
        if not rewards:
            raise ValueError("No reward values extracted from model response")

        # Check dimensions consistency
        if len(set(len(row) for row in rewards)) > 1:
            raise ValueError("Inconsistent row lengths in reward matrix")

        # Validate against grid size from JSON
        if len(rewards) != rows:
            raise ValueError(
                f"Grid size mismatch: JSON specifies {rows} rows, but got {len(rewards)}"
            )
        if len(rewards[0]) != cols:
            raise ValueError(
                f"Grid size mismatch: JSON specifies {cols} columns, but got {len(rewards[0])}"
            )

        # Validate against expected grid size if provided
        if expected_grid_size:
            expected_rows, expected_cols = expected_grid_size
            if len(rewards) != expected_rows:
                raise ValueError(
                    f"Expected {expected_rows} rows, but got {len(rewards)}"
                )
            if len(rewards[0]) != expected_cols:
                raise ValueError(
                    f"Expected {expected_cols} columns, but got {len(rewards[0])}"
                )

        return rewards



    def run_batch(
        self,
        image_paths: List[str],
        grid_size: Optional[Tuple[int, int]] = None
    ) -> List[List[List[float]]]:
        """
        Evaluate reward values for multiple map images.

        Args:
            image_paths: List of paths to map image files
            grid_size: Optional tuple of (rows, cols) if known

        Returns:
            A list of 2D reward matrices, one for each input image
        """
        results = []
        for image_path in image_paths:
            rewards = self.run(image_path, grid_size)
            results.append(rewards)
        return results





if __name__ == "__main__":

    main()