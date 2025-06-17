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
    Parses the YAML output from 'clang-format --dump-config' to identify options and their types.
    Handles nested dictionaries recursively.

    Args:
        yaml_string (str): The YAML output string.

    Returns:
        dict: A dictionary mapping option names to a dictionary containing
              'type' (str: 'bool', 'int', 'str', 'list', 'dict', etc.) and
              'value' (any: the parsed value).
              Returns None if parsing fails.
    """
    try:
        config = yaml.safe_load(yaml_string)
        if not isinstance(config, dict):
            print("Error: Parsed clang-format config is not a dictionary.", file=sys.stderr)
            return None

        # Helper function to recursively process the dictionary
        def process_dict(data):
            options_info = {}
            if not isinstance(data, dict):
                 # This case should ideally not happen if the top level is a dict,
                 # but handles potential malformed nested structures.
                 return data # Return the non-dict value as is

            for key, value in data.items():
                value_type = type(value).__name__
                if value_type == 'dict':
                    # Recurse into nested dictionary
                    options_info[key] = {'type': value_type, 'value': process_dict(value)}
                else:
                    # Store simple value and its type
                    options_info[key] = {'type': value_type, 'value': value}
            return options_info

        return process_dict(config)

    except yaml.YAMLError as e:
        print(f"Error parsing YAML output from clang-format: {e}", file=sys.stderr)
        return None

def generate_clang_format_config(options_info):
    """
    Generates a YAML string for a .clang-format file from a dictionary of options.
    Handles nested dictionaries recursively.

    Args:
        options_info (dict): A dictionary mapping option names to their info dicts
                             (containing 'type' and 'value'). Can be nested.

    Returns:
        str: A YAML formatted string.
    """
    # Helper function to recursively extract values
    def extract_values(data):
        if not isinstance(data, dict):
            return data # Return the simple value

        config_dict = {}
        for key, info in data.items():
            # Assuming info is a dict {'type': ..., 'value': ...}
            if info['type'] == 'dict':
                # Recurse into nested dictionary
                config_dict[key] = extract_values(info['value'])
            else:
                # Extract the simple value
                config_dict[key] = info['value']
        return config_dict

    # Start extraction from the root
    config_dict_values = extract_values(options_info)

    # Dump the extracted dictionary to YAML
    return yaml.dump(config_dict_values, default_flow_style=False, sort_keys=False)

