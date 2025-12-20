"""
Prompts for reward evaluation using vision language models.
"""

REWARD_EVALUATION_SYSTEM_PROMPT = """You are an expert at analyzing grid-based navigation maps and assigning reward values based on distance from goals.
You understand spatial relationships and can accurately evaluate optimal paths.
When you return results, you MUST respond with a single valid JSON object that follows the caller's requested schema, with no extra commentary, no markdown, and no code fences.
"""

def REWARD_EVALUATION_PROMPT(discount_factor: float = 0.9, grid_size: tuple = (4, 4)) -> str:
    """
    Generate the reward evaluation prompt with the specified discount factor.

    Args:
        discount_factor: The discount factor to use for reward propagation (default: 0.9)
        grid_size: Tuple of (rows, cols) representing the grid dimensions

    Returns:
        The formatted prompt string
    """
    rows, cols = grid_size
    return f"""Analyze the grid map shown in [Image 1]. This map contains:
- An agent (character sprite) at the starting position
- A goal (treasure/target) at the destination
- Obstacles (ice blocks or barriers) that cannot be traversed
- Empty cells (passable terrain)

Your task is to assign a reward value to each cell in the grid. The reward should:
1. Be HIGHEST at the goal cell (set the value to 1.0)
2. Decrease as distance from the goal increases (apply discounting with factor {discount_factor})
   - For cells at distance d from the goal, the reward should be approximately {discount_factor}^d
3. Be LOWEST or NEGATIVE for obstacle cells (set the value to -1.0)
4. Consider the shortest path distance, not straight-line distance
5. Cells that are unreachable from the goal should have very low or negative rewards

You MUST return the result in **JSON format**, as a single valid JSON object and nothing else.
The JSON must contain:
- The grid size (number of rows and columns)
- The reward values as a 2D matrix (a JSON list of lists of numbers)
- The grid size MUST BE "{rows} x {cols}"

Use the following schema:

{{
  "grid_size": {{
    "rows": {rows},
    "cols": {cols}
  }},
  "rewards": [
    [r_0_0, r_0_1, ..., r_0_({cols}-1)],
    [r_1_0, r_1_1, ..., r_1_({cols}-1)],
    ...
    [r_({rows}-1)_0, ..., r_({rows}-1)_({cols}-1)]
  ]
}}

Constraints:
- "rewards" MUST be a JSON list of lists (2D array) of numeric values (not strings).
- The outer "rewards" list length MUST equal {rows}.
- Each inner list length MUST equal {cols}.
- The total number of scalar elements in "rewards" MUST therefore be {rows} * {cols}.
- The goal cell reward MUST be exactly 1.0.
- Obstacle cells MUST be exactly -1.0.
- Use decimal numbers for all rewards (e.g., 0.0, -0.5, 1.0).

IMPORTANT:
- Do NOT include any fields other than "grid_size" and "rewards".
- Do NOT include any explanations, comments, markdown, or code fences.
- Your entire response MUST be exactly one valid JSON object following the schema above.
"""
