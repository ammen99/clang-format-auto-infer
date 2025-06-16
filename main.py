import argparse
import os
import subprocess
import sys
import yaml

def get_clang_format_options():
    """
    Runs 'clang-format --dump-config' to get a list of all possible options.

    Returns:
        str: The output from 'clang-format --dump-config'.
        None: If clang-format is not found or an error occurs.
    """
    try:
        result = subprocess.run(
            ["clang-format", "--dump-config"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except FileNotFoundError:
        print("Error: clang-format command not found. Please ensure it is installed and in your PATH.", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running clang-format: {e}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        return None

def parse_clang_format_options(yaml_string):
    """
    Parses the YAML output from 'clang-format --dump-config' to identify options and their types.

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

        options_info = {}
        for key, value in config.items():
            value_type = type(value).__name__
            options_info[key] = {'type': value_type, 'value': value}

        return options_info

    except yaml.YAMLError as e:
        print(f"Error parsing YAML output from clang-format: {e}", file=sys.stderr)
        return None


def main():
    """
    Parses command-line arguments for the clang-format optimization tool.
    """
    parser = argparse.ArgumentParser(
        description="Optimize clang-format configuration for a git repository."
    )
    parser.add_argument(
        "repo_path",
        help="Path to the git repository to analyze."
    )
    parser.add_argument(
        "--output",
        dest="output_file",
        help="Path to the file where the optimized configuration will be written (optional). If not provided, output is written to stdout."
    )

    args = parser.parse_args()

    # Basic validation (optional but good practice)
    if not os.path.isdir(args.repo_path):
        print(f"Error: Repository path '{args.repo_path}' is not a valid directory.", file=sys.stderr)
        exit(1)

    print(f"Analyzing repository: {args.repo_path}")

    options_output = get_clang_format_options()

    if not options_output:
        print("\nFailed to retrieve clang-format options.", file=sys.stderr)
        exit(1)

    options_info = parse_clang_format_options(options_output)

    if not options_info:
        print("\nFailed to parse clang-format options.", file=sys.stderr)
        exit(1)

    print(f"\nSuccessfully parsed {len(options_info)} clang-format options.")
    # print("Parsed options (first 5):")
    # for i, (key, info) in enumerate(list(options_info.items())[:5]):
    #     print(f"  {key}: type={info['type']}, value={info['value']}")

    # TODO: Add the main logic for configuration optimization using these options
    # For now, just handle writing the retrieved options (the original YAML)

    if args.output_file:
        print(f"Writing configuration to: {args.output_file}")
        try:
            # We write the original YAML output, not the parsed structure,
            # as the goal is to output a valid config file.
            with open(args.output_file, "w") as f:
                f.write(options_output)
            print("Configuration written successfully.")
        except IOError as e:
            print(f"Error writing to file {args.output_file}: {e}", file=sys.stderr)
            exit(1)
    else:
        print("Writing configuration to stdout:")
        # We write the original YAML output, not the parsed structure.
        print(options_output)


if __name__ == "__main__":
    main()
