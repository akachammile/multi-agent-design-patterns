from langchain.tools import tool


@tool
def load_skill(skill_name: str) -> str:
    """Load a specialized skill prompt.

    Available skills:
    - docx: docx generate、edit expert
    - pdf: pdf generate、edit expert

    Returns the skill's prompt and context.
    """
    # Load skill content from file/database
    ...
