import yaml
import sys
import subprocess
import copy
from .utils import run_command # Import run_command from utils

def get_clang_format_options(debug=False):
    """
    Runs 'clang-format --dump-config' to get a list of all possible options.

    Args:
        debug (bool): Enable debug output for run_command.

    Returns:
        str: The output from 'clang-format --dump-config'.
        None: If clang-format is not found or an error occurs.
    """
    cmd = ["clang-format", "--dump-config"]
    try:
        result = run_command(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            debug=debug # Pass debug flag
        )
        return result.stdout
    except FileNotFoundError:
        print("Error: clang-format command not found. Please ensure it is installed and in your PATH.", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running clang-format --dump-config: {e}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        return None

def parse_clang_format_options(yaml_string):
    """
    Parses the YAML output from 'clang-format --dump-config' and flattens it.
    Nested dictionaries are represented with dot-separated keys (e.g., "Parent.SubOption").

    Args:
        yaml_string (str): The YAML output string.

    Returns:
        dict: A flat dictionary mapping dot-separated option names to a dictionary containing
              'type' (str: 'bool', 'int', 'str', 'list', 'dict', etc.) and
              'value' (any: the parsed value).
              Returns None if parsing fails.
    """
    try:
        config = yaml.safe_load(yaml_string)
        if not isinstance(config, dict):
            print("Error: Parsed clang-format config is not a dictionary.", file=sys.stderr)
            return None

        flat_options = {}

        def flatten_dict(data, current_path=""):
            if not isinstance(data, dict):
                # This case should ideally not happen for the top-level config,
                # but handles potential malformed nested structures where a dict key
                # might map to a non-dict value unexpectedly.
                return

            for key, value in data.items():
                full_path = f"{current_path}.{key}" if current_path else key
                value_type = type(value).__name__
                if value_type == 'dict':
                    flatten_dict(value, full_path) # Recurse for nested dicts
                else:
                    flat_options[full_path] = {'type': value_type, 'value': value}

        flatten_dict(config)
        return flat_options

    except yaml.YAMLError as e:
        print(f"Error parsing YAML output from clang-format: {e}", file=sys.stderr)
        return None

def generate_clang_format_config(flat_options_info):
    """
    Generates a YAML string for a .clang-format file from a flat dictionary of options.
    Reconstructs nested dictionaries from dot-separated names.

    Args:
        flat_options_info (dict): A flat dictionary mapping dot-separated option names
                                  to their info dicts (containing 'type' and 'value').

    Returns:
        str: A YAML formatted string.
    """
    nested_config = {}

    # Sort keys to ensure consistent output order, especially for nested structures
    # This helps with reproducibility of the generated YAML.
    for full_path in sorted(flat_options_info.keys()):
        info = flat_options_info[full_path]
        parts = full_path.split('.')
        current_level = nested_config
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # This is the final part, assign the value
                current_level[part] = info['value']
            else:
                # This is an intermediate part, ensure it's a dictionary
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]
            
    return yaml.dump(nested_config, default_flow_style=False, sort_keys=False)

