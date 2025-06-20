import os
import sys
import tempfile
import shutil
import copy
import subprocess # For git commands in create_dummy_repo

# Import necessary modules from src
# Assuming test.py is at the project root, so relative imports from src work
from src.config_loader import load_json_option_values, load_forced_options
from src.clang_format_parser import get_clang_format_options, parse_clang_format_options, generate_clang_format_config
from src.repo_formatter import run_clang_format_and_count_changes
from src.optimizer import optimize_option_with_values # Import the specific function
from src.data_classes import IslandEvolutionArgs, GeneticAlgorithmLookups, WorkerContext # New imports for data classes

# Fixed random seed for reproducibility of file sampling in tests
RANDOM_SEED = 42

def create_dummy_repo(base_path):
    """Creates a dummy git repository with a C++ file for testing."""
    repo_dir = os.path.join(base_path, "test_repo_dummy")
    os.makedirs(repo_dir, exist_ok=True)

    try:
        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True, text=True)

        # Create a dummy C++ file
        cpp_content = """
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
"""
        cpp_file_path = os.path.join(repo_dir, "test.cpp")
        with open(cpp_file_path, "w") as f:
            f.write(cpp_content)

        # Add and commit the file
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, check=True, capture_output=True, text=True)

        print(f"Dummy repository created at: {repo_dir}", file=sys.stderr)
        return repo_dir
    except subprocess.CalledProcessError as e:
        print(f"Error creating dummy repo: {e.stderr}", file=sys.stderr)
        shutil.rmtree(repo_dir) # Clean up if creation fails
        sys.exit(1)
    except FileNotFoundError:
        print("Error: git command not found. Please ensure Git is installed and in your PATH.", file=sys.stderr)
        shutil.rmtree(repo_dir)
        sys.exit(1)


def main():
    # Create a temporary directory for the dummy repo
    with tempfile.TemporaryDirectory(prefix='clang_opt_test_') as tmp_base_dir:
        test_repo_path = create_dummy_repo(tmp_base_dir)

        # Define paths to config files relative to the project root
        script_dir = os.path.dirname(__file__)
        json_values_path = os.path.join(script_dir, "data", "clang-format-values.json")
        forced_options_path = os.path.join(script_dir, "data", "forced.yml")

        # Load configurations
        print(f"Loading JSON option values from: {json_values_path}", file=sys.stderr)
        json_options_lookup = load_json_option_values(json_values_path)
        print(f"Loading forced options from: {forced_options_path}", file=sys.stderr)
        forced_options_lookup = load_forced_options(forced_options_path)

        # Get base clang-format options (default dump-config)
        print("\nGetting base clang-format options...", file=sys.stderr)
        base_options_output = get_clang_format_options(debug=True)
        if not base_options_output:
            print("Failed to get base clang-format options. Exiting.", file=sys.stderr)
            sys.exit(1)
        base_options_info = parse_clang_format_options(base_options_output)
        if not base_options_info:
            print("Failed to parse base clang-format options. Exiting.", file=sys.stderr)
            sys.exit(1)
        print("Base clang-format options loaded.", file=sys.stderr)

        print("\n--- Starting individual option value tests ---", file=sys.stderr)
        total_options_tested = 0
        total_options_skipped = 0

        # --- Specific option to test ---
        option_to_test = "DerivePointerAlignment"

        if option_to_test not in json_options_lookup:
            print(f"Error: Option '{option_to_test}' not found in the JSON option values file. Exiting.", file=sys.stderr)
            sys.exit(1)

        option_name = option_to_test
        json_info = json_options_lookup[option_name]

        # Create GeneticAlgorithmLookups object for IslandEvolutionArgs
        dummy_lookups = GeneticAlgorithmLookups(
            json_options_lookup=json_options_lookup,
            forced_options_lookup=forced_options_lookup
        )

        # Create IslandEvolutionArgs for the test (representing a single island)
        test_island_args = IslandEvolutionArgs(
            population=[], # Not relevant for this single call
            island_population_size=0, # Not relevant
            island_index=0, # Assign island index 0 for test
            lookups=dummy_lookups,
            debug=True, # Use True for debug as per original call
            file_sample_percentage=100.0, # Default for test
            random_seed=RANDOM_SEED # Use the defined random seed
        )

        # Create a WorkerContext for the test (representing the main process as worker 1)
        test_worker_context = WorkerContext(
            repo_path=test_repo_path,
            process_id=1 # Main process can be considered worker 1 for testing
        )

        if option_name in forced_options_lookup:
            print(f"\nSkipping forced option: {option_name}", file=sys.stderr)
            total_options_skipped += 1
        elif option_name not in base_options_info:
            print(f"\nSkipping option '{option_name}': Not found in clang-format --dump-config output.", file=sys.stderr)
            total_options_skipped += 1
        else:
            possible_values = json_info['possible_values']
            if not possible_values:
                if base_options_info.get(option_name, {}).get('type') == 'bool':
                    possible_values = [True, False]
                else:
                    print(f"\nSkipping option '{option_name}': No possible values defined in JSON and not a boolean.", file=sys.stderr)
                    total_options_skipped += 1
            else:
                print(f"\n--- Testing option: {option_name} (Type: {json_info['type']}) ---", file=sys.stderr)
                print(f"  Possible values to test: {possible_values}", file=sys.stderr)

                # Create a fresh deep copy of base_options_info for each option being tested
                # This ensures that each option's test starts from a clean slate of default values.
                test_options_config = copy.deepcopy(base_options_info)

                original_value = test_options_config[option_name]['value']
                print(f"  Original value from dump-config: {original_value}", file=sys.stderr)

                # Call optimize_option_with_values to find the best value for this single option.
                # This function modifies `test_options_config` in place.
                # It also prints its own progress and results for each value it tests.
                optimize_option_with_values(
                    test_options_config,
                    option_name,
                    possible_values,
                    test_island_args, # Pass the IslandEvolutionArgs object
                    test_worker_context # Pass the WorkerContext object
                )

                # After optimize_option_with_values completes, `test_options_config` holds the
                # configuration with the 'best' value found for `option_name`.
                # We can now evaluate the fitness of this resulting configuration.
                final_value_after_optimization = test_options_config[option_name]['value']
                final_config_string = generate_clang_format_config(test_options_config)
                
                # Update the call to run_clang_format_and_count_changes
                final_changes = run_clang_format_and_count_changes(
                    final_config_string,
                    repo_path=test_worker_context.repo_path,
                    process_id=test_worker_context.process_id,
                    debug=test_island_args.debug,
                    file_sample_percentage=test_island_args.file_sample_percentage,
                    random_seed=test_island_args.random_seed
                )

                print(f"  Summary for '{option_name}':", file=sys.stderr)
                print(f"    Original value: {original_value}", file=sys.stderr)
                print(f"    Optimized value found: {final_value_after_optimization}", file=sys.stderr)
                print(f"    Final changes with optimized value: {final_changes}", file=sys.stderr)
                total_options_tested += 1

        print(f"\n--- Individual option value tests complete ---", file=sys.stderr)
        print(f"Total options tested: {total_options_tested}", file=sys.stderr)
        print(f"Total options skipped: {total_options_skipped}", file=sys.stderr)

if __name__ == "__main__":
    main()
