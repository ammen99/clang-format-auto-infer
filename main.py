import argparse
import os
import subprocess
import sys

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

    # TODO: Add the main logic for configuration optimization using these options
    # For now, just handle writing the retrieved options

    if args.output_file:
        print(f"Writing configuration to: {args.output_file}")
        try:
            with open(args.output_file, "w") as f:
                f.write(options_output)
            print("Configuration written successfully.")
        except IOError as e:
            print(f"Error writing to file {args.output_file}: {e}", file=sys.stderr)
            exit(1)
    else:
        print("Writing configuration to stdout:")
        print(options_output)


if __name__ == "__main__":
    main()
