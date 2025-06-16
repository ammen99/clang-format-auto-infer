import argparse
import os
import subprocess
import sys
import yaml
import re
import copy

# Global debug flag
DEBUG = False

# Dictionary of options that should have a fixed, forced value and not be optimized.
# Format: { "OptionName": ForcedValue, ... }
# Assumes OptionName is globally unique even if nested in the YAML structure.
FORCED_OPTIONS = {
    "DisableFormat": False, # We never want to disable formatting entirely
    # Add other options here if needed, e.g.,
    # "UseTab": "Never",
    # "IndentWidth": 4,
}


def run_command(cmd, capture_output=False, text=False, check=False, cwd=None):
    """
    Runs a subprocess command and optionally prints it if debug is enabled.

    Args:
        cmd (list): The command and its arguments.
        capture_output (bool): Whether to capture stdout/stderr.
        text (bool): Whether to decode stdout/stderr as text.
        check (bool): If True, raise CalledProcessError on non-zero exit code.
        cwd (str, optional): The working directory for the command.

    Returns:
        subprocess.CompletedProcess: The result of the subprocess run.
    """
    if DEBUG:
        cmd_str = ' '.join(cmd)
        cwd_str = f" (cwd: {cwd})" if cwd else ""
        print(f"Executing command: {cmd_str}{cwd_str}", file=sys.stderr)

    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=text,
            check=check,
            cwd=cwd
        )
        if DEBUG and capture_output:
             print(f"Command stdout:\n{result.stdout}", file=sys.stderr)
             print(f"Command stderr:\n{result.stderr}", file=sys.stderr)
        return result
    except FileNotFoundError:
        print(f"Error: Command not found: {cmd[0]}", file=sys.stderr)
        raise # Re-raise the exception
    except subprocess.CalledProcessError as e:
        if DEBUG:
             print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
             print(f"Stderr: {e.stderr}", file=sys.stderr)
        raise # Re-raise the exception


def get_clang_format_options():
    """
    Runs 'clang-format --dump-config' to get a list of all possible options.

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
            check=True
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
    # Ensure we are in the repo directory for git commands
    # os.chdir(repo_path) # Moved chdir into the try block for better cleanup

    temp_config_file = os.path.join(repo_path, ".clang-format.tmp") # Use a temp file inside the repo
    try:
        # Change to repo directory
        os.chdir(repo_path)

        # Write the configuration to the temporary file path inside the repo
        with open(temp_config_file, 'w') as tmp_file:
            tmp_file.write(config_string)

        # Find files to format (common C/C++ extensions)
        # Use git ls-files to only format tracked files
        git_ls_files_cmd = ["git", "ls-files", "--", "*.c", "*.cc", "*.cpp", "*.cxx", "*.h", "*.hh", "*.hpp", "*.hxx", "*.m", "*.mm"]
        try:
            result = run_command(git_ls_files_cmd, capture_output=True, text=True, check=True) # cwd is already repo_path
            files_to_format = result.stdout.splitlines()
        except subprocess.CalledProcessError as e:
            print(f"Error listing files in repo: {e}", file=sys.stderr)
            return -1

        if not files_to_format:
            if DEBUG:
                 print("No C/C++/Objective-C files found in the repository to format.", file=sys.stderr)
            return 0 # No files to format means no changes

        # Run clang-format on the files, explicitly using the temporary config file
        # We need to provide the full path to the files relative to the repo root
        # clang-format expects paths relative to the current directory, which is repo_path
        clang_format_cmd = ["clang-format", f"-style=file:{temp_config_file}", "-i"] + files_to_format
        try:
            # check=False because it might exit non-zero on formatting errors
            run_command(clang_format_cmd, check=False, capture_output=True, text=True) # cwd is already repo_path
        except FileNotFoundError:
            print("Error: clang-format command not found. Please ensure it is installed and in your PATH.", file=sys.stderr)
            return -1
        except Exception as e:
            print(f"Error running clang-format: {e}", file=sys.stderr)
            return -1

        # Count changes using git diff --shortstat from the repo directory
        git_diff_cmd = ["git", "diff", "--shortstat"]
        try:
            result = run_command(git_diff_cmd, capture_output=True, text=True, check=True) # cwd is already repo_path
            diff_output = result.stdout.strip()

            # Parse the output, e.g., " 1 file changed, 2 insertions(+), 2 deletions(-)"
            # We want the total of insertions and deletions
            total_changes = 0
            insertions_match = re.search(r'(\d+) insertions?\(\+\)', diff_output)
            deletions_match = re.search(r'(\d+) deletions?\(-\)', diff_output)

            if insertions_match:
                total_changes += int(insertions_match.group(1))
            if deletions_match:
                total_changes += int(deletions_match.group(1))

            if not insertions_match and not deletions_match and diff_output:
                 # If there's output but no insertions/deletions matched, something is unexpected
                 # This might happen if only file modes or whitespace changes outside lines are reported
                 if DEBUG:
                     print(f"Warning: Could not parse insertions/deletions from diff output: '{diff_output}'", file=sys.stderr)
                 # In this case, assume 0 line changes for optimization purposes
                 total_changes = 0
            elif not diff_output:
                 # No diff output means no changes
                 total_changes = 0


        except subprocess.CalledProcessError as e:
            print(f"Error running git diff: {e}", file=sys.stderr)
            return -1

        return total_changes

    finally:
        # Ensure we are in the repo directory before resetting
        if os.getcwd() != repo_path:
             # This shouldn't happen if chdir inside try works, but as a safeguard
             try:
                 os.chdir(repo_path)
             except OSError as e:
                 print(f"Error changing back to repo directory {repo_path} for cleanup: {e}", file=sys.stderr)
                 # Cannot proceed with git restore/file removal safely if we can't get to the repo dir
                 # Consider exiting or raising here depending on desired robustness
                 pass # Continue cleanup attempts if possible

        # Reset the repository changes
        git_restore_cmd = ["git", "restore", "."]
        try:
            run_command(git_restore_cmd, check=True, capture_output=True, text=True) # cwd is already repo_path
        except subprocess.CalledProcessError as e:
            print(f"Error resetting git repository: {e}", file=sys.stderr)
            # Note: Returning -1 here might mask the actual clang-format change count
            # Consider logging and proceeding, or raising an exception.
            # For now, we'll just print the error.

        # Clean up the temporary config file
        if os.path.exists(temp_config_file):
            try:
                os.remove(temp_config_file)
            except OSError as e:
                print(f"Error removing temporary config file {temp_config_file}: {e}", file=sys.stderr)

        # Change back to the original directory
        if os.getcwd() != original_cwd:
             try:
                 os.chdir(original_cwd)
             except OSError as e:
                 print(f"Error changing back to original directory {original_cwd}: {e}", file=sys.stderr)
                 # This is a significant issue, the script might leave the user in the wrong directory
                 # Consider exiting or raising here.
                 pass # Continue script execution


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


def optimize_boolean_option(parent_dict, option_name, repo_path, root_options_dict):
    """
    Optimizes a single boolean option by testing True and False values.

    Args:
        parent_dict (dict): The dictionary containing the option being optimized.
                            This could be the root dict or a nested dict.
        option_name (str): The name of the boolean option.
        repo_path (str): Path to the git repository.
        root_options_dict (dict): The top-level dictionary containing all options.
                                  Needed to generate the full config for testing.
    """
    option_info = parent_dict[option_name]
    original_value = option_info['value'] # Store original value in case of errors

    # Check if the option is in the forced list
    # Assumes option_name is globally unique
    if option_name in FORCED_OPTIONS:
        forced_value = FORCED_OPTIONS[option_name]
        print(f"\nSkipping optimization for '{option_name}'. Forcing value to: {forced_value}", file=sys.stderr)
        parent_dict[option_name]['value'] = forced_value
        return # Skip optimization for forced options

    print(f"\nOptimizing '{option_name}' (current: {original_value})...", file=sys.stderr)

    # --- Test False ---
    parent_dict[option_name]['value'] = False
    config_false = generate_clang_format_config(root_options_dict)
    print(f"  Testing '{option_name}: False'...", file=sys.stderr)
    changes_false = run_clang_format_and_count_changes(repo_path, config_false)
    if changes_false != -1:
        print(f"    Changes with False: {changes_false}", file=sys.stderr)
    else:
        print(f"    Error testing False.", file=sys.stderr)

    # --- Test True ---
    parent_dict[option_name]['value'] = True
    config_true = generate_clang_format_config(root_options_dict)
    print(f"  Testing '{option_name}: True'...", file=sys.stderr)
    changes_true = run_clang_format_and_count_changes(repo_path, config_true)
    if changes_true != -1:
         print(f"    Changes with True: {changes_true}", file=sys.stderr)
    else:
         print(f"    Error testing True.", file=sys.stderr)

    # --- Decide Best Value ---
    best_value = original_value # Default to original value
    best_changes = float('inf')

    # Case 1: Both tests failed
    if changes_false == -1 and changes_true == -1:
        print(f"  Both tests failed for '{option_name}'. Keeping original value: {original_value}", file=sys.stderr)
        # Value is currently True from the last test, restore original
        parent_dict[option_name]['value'] = original_value
        return # Optimization failed for this option

    # Case 2: Only False test succeeded
    if changes_false != -1 and changes_true == -1:
        best_value = False
        best_changes = changes_false
        print(f"  True test failed for '{option_name}'. Choosing False (changes: {best_changes})", file=sys.stderr)
    # Case 3: Only True test succeeded
    elif changes_false == -1 and changes_true != -1:
        best_value = True
        best_changes = changes_true
        print(f"  False test failed for '{option_name}'. Choosing True (changes: {best_changes})", file=sys.stderr)
    # Case 4: Both tests succeeded
    elif changes_false != -1 and changes_true != -1:
        if changes_false < changes_true:
            best_value = False
            best_changes = changes_false
            print(f"  False resulted in fewer changes for '{option_name}'. Choosing False (changes: {changes_false})", file=sys.stderr)
        elif changes_true < changes_false:
            best_value = True
            best_changes = changes_true
            print(f"  True resulted in fewer changes for '{option_name}'. Choosing True (changes: {changes_true})", file=sys.stderr)
        else: # changes_false == changes_true
            best_value = original_value # Keep original on a tie
            best_changes = changes_false # or changes_true
            print(f"  Changes are equal for '{option_name}' ({best_changes}). Keeping original value: {original_value}", file=sys.stderr)

    # Update the working dictionary with the chosen value (unless both failed, handled above)
    if changes_false != -1 or changes_true != -1:
         parent_dict[option_name]['value'] = best_value


def optimize_options_recursively(current_options_dict, repo_path, root_options_dict):
    """
    Recursively iterates through options in a dictionary structure and optimizes boolean ones.

    Args:
        current_options_dict (dict): The dictionary currently being processed
                                     (could be the root dict or a nested dict).
                                     This dictionary is modified in place.
        repo_path (str): Path to the git repository.
        root_options_dict (dict): The top-level dictionary containing all options.
                                  Needed to generate the full config for testing.
    """
    # Iterate over a list of keys to avoid issues modifying dict during iteration
    for option_name in list(current_options_dict.keys()):
        option_info = current_options_dict[option_name]

        if option_info['type'] == 'dict':
            # Recurse into nested dictionary
            if DEBUG:
                 print(f"Entering nested options for '{option_name}'...", file=sys.stderr)
            optimize_options_recursively(option_info['value'], repo_path, root_options_dict)
            if DEBUG:
                 print(f"Exiting nested options for '{option_name}'.", file=sys.stderr)
        elif option_info['type'] == 'bool':
            # Optimize boolean option
            optimize_boolean_option(current_options_dict, option_name, repo_path, root_options_dict)
        else:
            # Skip non-boolean, non-dict options
            if DEBUG:
                 print(f"Skipping non-boolean, non-dict option: '{option_name}' (type: {option_info['type']})", file=sys.stderr)


def main():
    """
    Parses command-line arguments and runs the clang-format optimization tool.
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
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug output (print commands being executed)."
    )

    args = parser.parse_args()

    # Set global debug flag
    global DEBUG
    DEBUG = args.debug

    # Basic validation
    if not os.path.isdir(args.repo_path):
        print(f"Error: Repository path '{args.repo_path}' is not a valid directory.", file=sys.stderr)
        exit(1)

    # Ensure the path is absolute for reliable chdir/restore
    repo_path_abs = os.path.abspath(args.repo_path)
    if not os.path.isdir(repo_path_abs):
         print(f"Error: Absolute repository path '{repo_path_abs}' is not a valid directory.", file=sys.stderr)
         exit(1)


    print(f"Analyzing repository: {repo_path_abs}")

    options_output = get_clang_format_options()

    if not options_output:
        print("\nFailed to retrieve clang-format options.", file=sys.stderr)
        exit(1)

    options_info = parse_clang_format_options(options_output)

    if not options_info:
        print("\nFailed to parse clang-format options.", file=sys.stderr)
        exit(1)

    # Note: The count from parse_clang_format_options is only top-level.
    # A more accurate count would require traversing the structure.
    print(f"\nSuccessfully parsed clang-format options structure.")

    # Create a working copy of options to optimize
    # This copy will be modified recursively by the optimization functions
    optimized_options_info = copy.deepcopy(options_info)

    print("\nStarting optimization...")

    # Start the recursive optimization process from the root
    optimize_options_recursively(optimized_options_info, repo_path_abs, optimized_options_info)

    print("\nOptimization complete.")

    # Generate the final optimized configuration from the modified structure
    optimized_config = generate_clang_format_config(optimized_options_info)

    # Output the final configuration
    if args.output_file:
        print(f"\nWriting optimized configuration to: {args.output_file}")
        try:
            with open(args.output_file, "w") as f:
                f.write(optimized_config)
            print("Optimized configuration written successfully.")
        except IOError as e:
            print(f"Error writing to file {args.output_file}: {e}", file=sys.stderr)
            exit(1)
    else:
        print("\nOptimized configuration:")
        print(optimized_config)


if __name__ == "__main__":
    main()
