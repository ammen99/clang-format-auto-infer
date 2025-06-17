import sys
import copy
from .repo_formatter import run_clang_format_and_count_changes # Import formatter
from .clang_format_parser import generate_clang_format_config # Import config generator

def optimize_option_with_values(flat_options_info, full_option_path, repo_path, possible_values, debug=False):
    """
    Optimizes a single option by testing each value in the provided list.

    Args:
        flat_options_info (dict): The flat dictionary containing all options.
                                  This dictionary is modified in place.
        full_option_path (str): The dot-separated full name of the option.
        repo_path (str): Path to the git repository.
        possible_values (list): A list of values to test for this option.
        debug (bool): Enable debug output.
    """
    option_info = flat_options_info[full_option_path]
    original_value = option_info['value'] # Store original value

    print(f"\nOptimizing '{full_option_path}' (current: {original_value})...", file=sys.stderr)
    print(f"  Testing values: {possible_values}", file=sys.stderr)

    min_changes = float('inf')
    best_value = original_value

    for value_to_test in possible_values:
        # Ensure the value type matches the expected type from dump-config
        # Simple type conversion for common types
        if option_info['type'] == 'bool':
            # Convert string 'true'/'false' to boolean
            if isinstance(value_to_test, str):
                if value_to_test.lower() == 'true':
                    value_to_test = True
                elif value_to_test.lower() == 'false':
                    value_to_test = False
                # else: keep as string, might be an enum value represented as string
            # Ensure boolean values are passed as bool, not strings 'True'/'False'
            elif isinstance(value_to_test, str) and value_to_test in ['True', 'False']:
                 value_to_test = value_to_test == 'True'


        elif option_info['type'] == 'int':
             try:
                 value_to_test = int(value_to_test)
             except (ValueError, TypeError):
                 print(f"Warning: Could not convert value '{value_to_test}' to int for option '{full_option_path}'. Skipping.", file=sys.stderr)
                 continue # Skip this value

        # Add other type conversions if necessary (e.g., float, list, etc.)
        # For now, assume other types (like strings for enums) can be used directly


        flat_options_info[full_option_path]['value'] = value_to_test
        config_string = generate_clang_format_config(flat_options_info) # Pass the flat dict

        # run_clang_format_and_count_changes will now exit on critical clang-format error,
        # return float('inf') on invalid config error, or return >= 0 on success, or -1 on git error.
        changes = run_clang_format_and_count_changes(repo_path, config_string, debug=debug)

        # We now consider float('inf') as a valid (but high) result, not an error to skip
        if changes != -1: # Only skip if it's a git-related error (-1)
            # Treat float('inf') as a very high change count
            print(f"    Changes with {value_to_test}: {changes}", file=sys.stderr)
            if changes < min_changes:
                min_changes = changes
                best_value = value_to_test
        else:
            # An error occurred in run_clang_format_and_count_changes (e.g., git diff failed)
            # The error message is already printed by that function.
            # We just need to skip this value and continue with the next one.
            pass # Error message already printed, continue loop


    # --- Decide Best Value ---
    # If min_changes is still infinity, it means all tests failed (returned -1 or float('inf'))
    # If all tests returned float('inf'), min_changes will be float('inf'), and best_value will be the last tested value.
    # If some tests returned float('inf') and some returned >= 0, min_changes will be the minimum >= 0.
    # If all tests returned -1, min_changes will be float('inf').
    if min_changes == float('inf'):
        # This happens if all tested values resulted in either a git error (-1) or an invalid clang-format config (float('inf')).
        # In this case, we keep the original value as we couldn't find a better valid one.
        print(f"  All tests failed or resulted in invalid configurations for '{full_option_path}'. Keeping original value: {original_value}", file=sys.stderr)
        # Value is currently the last tested value, restore original
        flat_options_info[full_option_path]['value'] = original_value
    else:
        # min_changes is a finite number (>= 0), meaning at least one configuration was valid and formatted files.
        print(f"  Best value for '{full_option_path}': {best_value} (changes: {min_changes})", file=sys.stderr)
        flat_options_info[full_option_path]['value'] = best_value


def optimize_all_options(flat_options_info, repo_path, json_options_lookup, forced_options_lookup, debug=False):
    """
    Iterates through all options in a flat dictionary structure and optimizes them
    using possible values from the JSON lookup if available.

    Args:
        flat_options_info (dict): The flat dictionary containing all options.
                                  This dictionary is modified in place.
        repo_path (str): Path to the git repository.
        json_options_lookup (dict): A dictionary mapping option names to their info
                                    from the JSON file, used to find possible values.
        forced_options_lookup (dict): A dictionary mapping option names to forced values.
        debug (bool): Enable debug output.
    """
    # Sort keys to ensure consistent optimization order (e.g., alphabetical)
    # This can be important if options interact, though generally clang-format options
    # are designed to be independent. Alphabetical is a reasonable default.
    sorted_option_paths = sorted(flat_options_info.keys())

    for full_option_path in sorted_option_paths:
        option_info = flat_options_info[full_option_path]

        # Check if the option is in the forced list
        if full_option_path in forced_options_lookup:
            forced_value = forced_options_lookup[full_option_path]
            print(f"\nSkipping optimization for '{full_option_path}'. Forcing value to: {forced_value}", file=sys.stderr)
            flat_options_info[full_option_path]['value'] = forced_value
            continue # Skip optimization for forced options

        # Check if we have possible values for this option from the JSON
        if full_option_path in json_options_lookup and json_options_lookup[full_option_path]['possible_values']:
            possible_values = json_options_lookup[full_option_path]['possible_values']
            optimize_option_with_values(flat_options_info, full_option_path, repo_path, possible_values, debug=debug)
        elif option_info['type'] == 'bool': # Check if it's a boolean from dump-config
             # If the option is not in the JSON lookup but is a boolean, test True/False
             print(f"\n'{full_option_path}' not found in JSON, but is boolean. Testing True/False.", file=sys.stderr)
             optimize_option_with_values(flat_options_info, full_option_path, repo_path, [True, False], debug=debug)
        else:
            # If no possible values from JSON and not a boolean, keep the default value
            if debug:
                 print(f"Skipping optimization for '{full_option_path}' (type: {option_info['type']}). No possible values provided in JSON for optimization and not a boolean.", file=sys.stderr)

