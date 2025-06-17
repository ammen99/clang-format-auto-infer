import json
import yaml
import os
import sys

def load_json_option_values(file_path):
    """
    Loads clang-format option values and possible values from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary mapping option names to their info (including possible_values).
              Returns an empty dict if file_path is None. Exits on error.
    """
    if not file_path:
        return {}

    if not os.path.exists(file_path):
        print(f"Error: Option values JSON file not found at '{file_path}'.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(file_path, 'r') as f:
            json_list = json.load(f)
            if not isinstance(json_list, list):
                print(f"Error: JSON file '{file_path}' does not contain a list.", file=sys.stderr)
                sys.exit(1)
            # Create a lookup dictionary by option name
            json_options_lookup = {item['name']: item for item in json_list if 'name' in item}
        print(f"Successfully loaded option values from '{file_path}'.", file=sys.stderr)
        return json_options_lookup
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading option values JSON file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)

def load_forced_options(file_path):
    """
    Loads forced clang-format options from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: A dictionary mapping option names to their forced values.
              Returns an empty dict if file_path is None. Exits on error.
    """
    if not file_path:
        return {}

    if not os.path.exists(file_path):
        print(f"Error: Forced options YAML file not found at '{file_path}'.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(file_path, 'r') as f:
            forced_options_lookup = yaml.safe_load(f)
            if not isinstance(forced_options_lookup, dict):
                 print(f"Error: YAML file '{file_path}' does not contain a dictionary.", file=sys.stderr)
                 sys.exit(1)
        print(f"Successfully loaded forced options from '{file_path}'.", file=sys.stderr)
        return forced_options_lookup
    except yaml.YAMLError as e:
        print(f"Error parsing YAML from '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading forced options YAML file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)

