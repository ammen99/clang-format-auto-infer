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

    Args:
        options_info (dict): A dictionary mapping option names to their info dicts
                             (containing 'type' and 'value').

    Returns:
        str: A YAML formatted string.
    """
    # We only need the values from the options_info dictionary
    config_dict = {key: info['value'] for key, info in options_info.items()}
    return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)


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

    print(f"\nSuccessfully parsed {len(options_info)} clang-format options.")

    # Create a working copy of options to optimize
    optimized_options_info = copy.deepcopy(options_info)

    print("\nStarting optimization for boolean options...")

    # Iterate through options and optimize boolean ones
    # Iterate over a list of keys to avoid issues modifying dict during iteration
    for option_name in list(options_info.keys()):
        option_info = options_info[option_name] # Get info from original to check type
        current_optimized_value = optimized_options_info[option_name]['value'] # Get current value from working copy

        # Check if the option is in the forced list
        if option_name in FORCED_OPTIONS:
            forced_value = FORCED_OPTIONS[option_name]
            print(f"\nSkipping optimization for '{option_name}'. Forcing value to: {forced_value}", file=sys.stderr)
            optimized_options_info[option_name]['value'] = forced_value
            continue # Skip to the next option

        if option_info['type'] == 'bool':
            print(f"\nOptimizing '{option_name}' (current: {current_optimized_value})...", file=sys.stderr)

            # --- Test False ---
            optimized_options_info[option_name]['value'] = False
            config_false = generate_clang_format_config(optimized_options_info)
            print(f"  Testing '{option_name}: False'...", file=sys.stderr)
            changes_false = run_clang_format_and_count_changes(repo_path_abs, config_false)
            if changes_false != -1:
                print(f"    Changes with False: {changes_false}", file=sys.stderr)
            else:
                print(f"    Error testing False.", file=sys.stderr)

            # --- Test True ---
            optimized_options_info[option_name]['value'] = True
            config_true = generate_clang_format_config(optimized_options_info)
            print(f"  Testing '{option_name}: True'...", file=sys.stderr)
            changes_true = run_clang_format_and_count_changes(repo_path_abs, config_true)
            if changes_true != -1:
                 print(f"    Changes with True: {changes_true}", file=sys.stderr)
            else:
                 print(f"    Error testing True.", file=sys.stderr)

            # --- Decide Best Value ---
            best_value = current_optimized_value # Default to current value in case of errors
            best_changes = float('inf')

            if changes_false != -1:
                best_changes = changes_false
                best_value = False

            # Use <= to prefer True if changes are equal, or if False test failed
            if changes_true != -1 and changes_true <= best_changes:
                 best_changes = changes_true
                 best_value = True

            # If both failed, keep original value (which is the default)
            if changes_false == -1 and changes_true == -1:
                 print(f"  Both tests failed for '{option_name}'. Keeping original value: {current_optimized_value}", file=sys.stderr)
                 # The value in optimized_options_info is already the last one tested (True),
                 # so we need to explicitly set it back to the original if both failed.
                 optimized_options_info[option_name]['value'] = current_optimized_value
            else:
                 # Update the working dictionary with the chosen value
                 optimized_options_info[option_name]['value'] = best_value
                 print(f"  Chosen value for '{option_name}': {best_value} (changes: {best_changes})", file=sys.stderr)

        # else:
            # print(f"Skipping non-boolean option: {option_name} (type: {option_info['type']})") # Too verbose for default run

    print("\nBoolean option optimization complete.")

    # Generate the final optimized configuration
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
