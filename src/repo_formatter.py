import os
import subprocess
import sys
import re
import shutil
import random # New import for file sampling
from .utils import run_command # Import run_command from utils

def run_clang_format_and_count_changes(repo_path, config_string, debug=False, file_sample_percentage=100.0, random_seed=None):
    """
    Runs a clang-format configuration on a repository, counts changes, and resets.

    Args:
        repo_path (str): Path to the git repository.
        config_string (str): The clang-format configuration as a YAML string.
        debug (bool): Enable debug output for run_command.
        file_sample_percentage (float): Percentage of files to randomly sample for formatting.
                                        Must be between 0.0 and 100.0.
        random_seed (int, optional): Seed for the random number generator used for file sampling.
                                     If None, a non-deterministic sample will be used.

    Returns:
        int or float('inf'): The total number of lines added or deleted by clang-format (>= 0).
                             Returns float('inf') if clang-format reports an invalid configuration
                             (e.g., "cannot be used with").
                             Returns -1 if a non-clang-format error occurs (like git diff or file listing).
                             Exits the script if a different type of clang-format execution error occurs.
    """
    original_cwd = os.getcwd()
    # Ensure we are in the repo directory for git commands
    # os.chdir(repo_path) # Moved chdir into the try block for better cleanup

    temp_config_file = os.path.join(repo_path, ".clang-format.tmp") # Use a temp file inside the repo
    error_config_dest = "/tmp/clang-format.yml" # Destination for error config file

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
            result = run_command(git_ls_files_cmd, capture_output=True, text=True, check=True, debug=debug) # cwd is already repo_path
            files_to_format = result.stdout.splitlines()
        except subprocess.CalledProcessError as e:
            print(f"Error listing files in repo: {e}", file=sys.stderr)
            return -1 # Indicate failure

        if not files_to_format:
            if debug:
                 print("No C/C++/Objective-C files found in the repository to format.", file=sys.stderr)
            return 0 # No files to format means no changes

        # Apply file sampling if percentage is less than 100%
        if file_sample_percentage < 100.0:
            num_files_to_sample = max(1, int(len(files_to_format) * (file_sample_percentage / 100.0)))
            if num_files_to_sample > len(files_to_format):
                num_files_to_sample = len(files_to_format) # Cap at total number of files

            if debug:
                print(f"  Sampling {num_files_to_sample} files ({file_sample_percentage:.1f}%) from {len(files_to_format)} available files.", file=sys.stderr)

            # Use a seeded random generator for reproducibility
            if random_seed is not None:
                rng = random.Random(random_seed)
                sampled_files = rng.sample(files_to_format, num_files_to_sample)
            else:
                # Fallback if no seed is provided (less reproducible)
                sampled_files = random.sample(files_to_format, num_files_to_sample)
            files_to_format = sampled_files
        
        # Run clang-format on the files, explicitly using the temporary config file
        # We need to provide the full path to the files relative to the repo root
        # clang-format expects paths relative to the current directory, which is repo_path
        clang_format_cmd = ["clang-format", f"-style=file:{temp_config_file}", "-i"] + files_to_format
        try:
            # check=True to catch clang-format errors
            run_command(clang_format_cmd, check=True, capture_output=True, text=True, debug=debug) # cwd is already repo_path
        except FileNotFoundError:
            print("Error: clang-format command not found. Please ensure it is installed and in your PATH.", file=sys.stderr)
            # Cannot copy config if clang-format isn't found to even try running it
            sys.exit(1) # Exit immediately as requested
        except subprocess.CalledProcessError as e:
            # Catch specific clang-format errors and report them
            error_output = (e.stdout or "") + (e.stderr or "") # Combine stdout and stderr for checking

            if "cannot be used with" in error_output:
                # This is a known invalid configuration error, treat as high cost
                print(f"Warning: clang-format reported an invalid configuration ('cannot be used with'). Treating as high cost.", file=sys.stderr)
                if debug:
                    print(f"Command: {' '.join(e.cmd)}", file=sys.stderr)
                    print(f"Exit code: {e.returncode}", file=sys.stderr)
                    if e.stdout: print(f"Stdout:\n{e.stdout}", file=sys.stderr)
                    if e.stderr: print(f"Stderr:\n{e.stderr}", file=sys.stderr)
                # Return infinity to signify a very bad configuration
                return float('inf')
            else:
                # Other clang-format errors are critical, exit
                print(f"Error running clang-format with the current configuration:", file=sys.stderr)
                print(f"Command: {' '.join(e.cmd)}", file=sys.stderr)
                print(f"Exit code: {e.returncode}", file=sys.stderr)
                if e.stdout:
                     print(f"Stdout:\n{e.stdout}", file=sys.stderr)
                if e.stderr:
                     print(f"Stderr:\n{e.stderr}", file=sys.stderr)

                # Copy the problematic config file before exiting
                try:
                    shutil.copyfile(temp_config_file, error_config_dest)
                    print(f"\nConfiguration causing the error copied to {error_config_dest} for inspection.", file=sys.stderr)
                except IOError as copy_error:
                    print(f"Error copying temporary config file to {error_config_dest}: {copy_error}", file=sys.stderr)

                sys.exit(1) # Exit immediately as requested

        # Count changes using git diff --shortstat from the repo directory
        git_diff_cmd = ["git", "diff", "--shortstat"]
        try:
            result = run_command(git_diff_cmd, capture_output=True, text=True, check=True, debug=debug) # cwd is already repo_path
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
                 if debug:
                     print(f"Warning: Could not parse insertions/deletions from diff output: '{diff_output}'", file=sys.stderr)
                 # In this case, assume 0 line changes for optimization purposes
                 total_changes = 0
            elif not diff_output:
                 # No diff output means no changes
                 total_changes = 0


        except subprocess.CalledProcessError as e:
            print(f"Error running git diff: {e}", file=sys.stderr)
            return -1 # Indicate failure

        return total_changes

    finally:
        # Ensure we are in the repo directory before resetting
        # This block runs even if sys.exit() is called, but cleanup might be incomplete
        # depending on the exact point of failure and OS.
        # The copy happens *before* exit, so that part is guaranteed.
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
            run_command(git_restore_cmd, check=True, capture_output=True, text=True, debug=debug) # cwd is already repo_path
        except subprocess.CalledProcessError as e:
            print(f"Error resetting git repository: {e}", file=sys.stderr)
            # Note: If clang-format failed and we exited, this might not be reached.
            # If git diff failed, this should still run.

        # Clean up the temporary config file
        # This file is copied *before* exit on clang-format error, so removing it here is fine.
        # It's also removed if clang-format succeeds or returns float('inf').
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
