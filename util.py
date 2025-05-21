import os
import yaml

def extract_code_from_markdown(markown_text: str, type_=None):
    """
    Extracts code blocks from markdown text.

    Args:
        markown_text (str): The markdown text.
        type_ (str): The type of code block to extract.

    Returns:
        list of str: The extracted code blocks.
    """
    code_blocks = []
    in_code_block = False
    code_block = []
    for line in markown_text.split("\n"):
        if line.strip().startswith("```"):
            if in_code_block:
                in_code_block = False
                code_blocks.append("\n".join(code_block))
                code_block = []
            else:
                if type_ is None or line.strip() == f"```{type_}":
                    in_code_block = True
        elif in_code_block:
            code_block.append(line)
    return code_blocks


def load_config(config_file=None):
    if config_file is None:
        config_file = 'config.yaml'

    base_path = os.path.dirname(__file__)  # Get current file directory
    config_file = os.path.join(base_path, config_file)

    with open(config_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)