import argparse
import os
import subprocess
import sys
import yaml
import tempfile
import re

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

def run_clang_format_and_count_changes(repo_path, config_string):
    """
    Applies a clang-format configuration to a repository, counts changes, and resets.

    Args:
        repo_path (str): Path to the git repository.
        config_string (str): The clang-format configuration as a YAML string.

    Returns:
        int: The total number of lines added or deleted by clang-format.
             Returns -1 if an error occurs.
    """
    original_cwd = os.getcwd()
    os.chdir(repo_path)

    temp_config_file = None
    try:
        # Create a temporary .clang-format file in the repo root
        # Use delete=False so clang-format can access it by path
        with tempfile.NamedTemporaryFile(mode='w', delete=False, prefix='.clang-format-', dir='.', suffix='.yaml') as tmp_file:
            tmp_file.write(config_string)
            temp_config_file = tmp_file.name

        # Find files to format (common C/C++ extensions)
        # Use git ls-files to only format tracked files
        git_ls_files_cmd = ["git", "ls-files", "--", "*.c", "*.cc", "*.cpp", "*.cxx", "*.h", "*.hh", "*.hpp", "*.hxx", "*.m", "*.mm"]
        try:
            result = subprocess.run(git_ls_files_cmd, capture_output=True, text=True, check=True)
            files_to_format = result.stdout.splitlines()
        except subprocess.CalledProcessError as e:
            print(f"Error listing files in repo: {e}", file=sys.stderr)
            return -1

        if not files_to_format:
            print("No C/C++/Objective-C files found in the repository to format.", file=sys.stderr)
            return 0 # No files to format means no changes

        # Run clang-format on the files
        clang_format_cmd = ["clang-format", "-style=file", "-i"] + files_to_format
        try:
            # Run in the background as it can be noisy, check=False because it might exit non-zero on formatting errors
            subprocess.run(clang_format_cmd, check=False, capture_output=True, text=True)
        except FileNotFoundError:
            print("Error: clang-format command not found. Please ensure it is installed and in your PATH.", file=sys.stderr)
            return -1
        except Exception as e:
            print(f"Error running clang-format: {e}", file=sys.stderr)
            return -1

        # Count changes using git diff --shortstat
        git_diff_cmd = ["git", "diff", "--shortstat"]
        try:
            result = subprocess.run(git_diff_cmd, capture_output=True, text=True, check=True)
            diff_output = result.stdout.strip()

            # Parse the output, e.g., " 1 file changed, 2 insertions(+), 2 deletions(-)"
            # We want the total of insertions and deletions
            match = re.search(r'(\d+) insertions?\(\+\).*?(\d+) deletions?\(-\)', diff_output)
            if match:
                insertions = int(match.group(1))
                deletions = int(match.group(2))
                total_changes = insertions + deletions
            else:
                 # Handle cases with only insertions or deletions, or no changes
                insertions_match = re.search(r'(\d+) insertions?\(\+\)', diff_output)
                deletions_match = re.search(r'(\d+) deletions?\(-\)', diff_output)
                total_changes = 0
                if insertions_match:
                    total_changes += int(insertions_match.group(1))
                if deletions_match:
                    total_changes += int(deletions_match.group(1))
                if not insertions_match and not deletions_match and diff_output:
                     # If there's output but no insertions/deletions matched, something is unexpected
                     print(f"Warning: Could not parse diff output: '{diff_output}'", file=sys.stderr)
                     # Attempt to count lines changed based on file changes reported by shortstat
                     file_change_match = re.search(r'(\d+) file changed', diff_output)
                     if file_change_match:
                         print("Estimating changes based on file count...", file=sys.stderr)
                         # This is a rough estimate, could be 1 line per file or more
                         # For simplicity, we might just return 0 or a warning value here
                         # Let's return 0 if no insertions/deletions are explicitly found
                         total_changes = 0
                     elif diff_output:
                          print(f"Warning: Unexpected diff output format: '{diff_output}'", file=sys.stderr)
                          total_changes = 0 # Assume 0 changes if format is unexpected


        except subprocess.CalledProcessError as e:
            print(f"Error running git diff: {e}", file=sys.stderr)
            return -1

        return total_changes

    finally:
        # Reset the repository changes
        try:
            subprocess.run(["git", "restore", "."], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error resetting git repository: {e}", file=sys.stderr)
            # Note: Returning -1 here might mask the actual clang-format change count
            # Consider logging and proceeding, or raising an exception.
            # For now, we'll just print the error.

        # Clean up the temporary config file
        if temp_config_file and os.path.exists(temp_config_file):
            os.remove(temp_config_file)

        # Change back to the original directory
        os.chdir(original_cwd)


def generate_clang_format_config(options_info):
    """
    Generates a YAML string for a .clang-format file from a dictionary of options.

    Args:
        options_info (dict): A dictionary mapping option names to their values.

    Returns:
        str: A YAML formatted string.
    """
    # We only need the values from the options_info dictionary
    config_dict = {key: info['value'] for key, info in options_info.items()}
    return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)


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
    # For now, generate and write the config from the parsed options

    generated_config = generate_clang_format_config(options_info)

    if args.output_file:
        print(f"Writing configuration to: {args.output_file}")
        try:
            with open(args.output_file, "w") as f:
                f.write(generated_config)
            print("Configuration written successfully.")
        except IOError as e:
            print(f"Error writing to file {args.output_file}: {e}", file=sys.stderr)
            exit(1)
    # TODO: Add the main logic for configuration optimization using these options
    # For now, generate and write the config from the parsed options

    generated_config = generate_clang_format_config(options_info)

    # Example usage of the new function (can be removed later)
    # print("\nRunning clang-format with generated config and counting changes...")
    # changes = run_clang_format_and_count_changes(args.repo_path, generated_config)
    # if changes != -1:
    #     print(f"Total lines changed by clang-format: {changes}")
    # else:
    #     print("Failed to count changes.")


    if args.output_file:
        print(f"Writing configuration to: {args.output_file}")
        try:
            with open(args.output_file, "w") as f:
                f.write(generated_config)
            print("Configuration written successfully.")
        except IOError as e:
            print(f"Error writing to file {args.output_file}: {e}", file=sys.stderr)
            exit(1)
    else:
        print("Writing configuration to stdout:")
        print(generated_config)


if __name__ == "__main__":
    main()
