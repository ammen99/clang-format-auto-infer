import sys
import copy
from .repo_formatter import run_clang_format_and_count_changes # Import formatter
from .clang_format_parser import generate_clang_format_config # Import config generator

def optimize_option_with_values(parent_dict, option_name, repo_path, root_options_dict, possible_values, debug=False):
    """
    Optimizes a single option by testing each value in the provided list.

    Args:
        parent_dict (dict): The dictionary containing the option being optimized.
                            This could be the root dict or a nested dict.
        option_name (str): The name of the option.
        repo_path (str): Path to the git repository.
        root_options_dict (dict): The top-level dictionary containing all options.
                                  Needed to generate the full config for testing.
        possible_values (list): A list of values to test for this option.
        debug (bool): Enable debug output.
    """
    option_info = parent_dict[option_name]
    original_value = option_info['value'] # Store original value

    print(f"\nOptimizing '{option_name}' (current: {original_value})...", file=sys.stderr)
    print(f"  Testing values: {possible_values}", file=sys.stderr)

    min_changes = float('inf')
    best_value = original_value
    # results = {} # Store results for printing - removed as not currently used

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
                 print(f"Warning: Could not convert value '{value_to_test}' to int for option '{option_name}'. Skipping.", file=sys.stderr)
                 # results[value_to_test] = -1 # Mark as failed - removed
                 continue # Skip this value

        # Add other type conversions if necessary (e.g., float, list, etc.)
        # For now, assume other types (like strings for enums) can be used directly


        parent_dict[option_name]['value'] = value_to_test
        config_string = generate_clang_format_config(root_options_dict)

        # run_clang_format_and_count_changes will now exit on critical clang-format error,
        # return float('inf') on invalid config error, or return >= 0 on success, or -1 on git error.
        changes = run_clang_format_and_count_changes(repo_path, config_string, debug=debug)
        # results[value_to_test] = changes # removed

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
        print(f"  All tests failed or resulted in invalid configurations for '{option_name}'. Keeping original value: {original_value}", file=sys.stderr)
        # Value is currently the last tested value, restore original
        parent_dict[option_name]['value'] = original_value
    else:
        # min_changes is a finite number (>= 0), meaning at least one configuration was valid and formatted files.
        print(f"  Best value for '{option_name}': {best_value} (changes: {min_changes})", file=sys.stderr)
        parent_dict[option_name]['value'] = best_value


def optimize_options_recursively(current_options_dict, repo_path, root_options_dict, json_options_lookup, forced_options_lookup, debug=False):
    """
    Recursively iterates through options in a dictionary structure and optimizes them
    using possible values from the JSON lookup if available.

    Args:
        current_options_dict (dict): The dictionary currently being processed
                                     (could be the root dict or a nested dict).
                                     This dictionary is modified in place.
        repo_path (str): Path to the git repository.
        root_options_dict (dict): The top-level dictionary containing all options.
                                  Needed to generate the full config for testing.
        json_options_lookup (dict): A dictionary mapping option names to their info
                                    from the JSON file, used to find possible values.
        forced_options_lookup (dict): A dictionary mapping option names to forced values.
        debug (bool): Enable debug output.
    """
    # Iterate over a list of keys to avoid issues modifying dict during iteration
    for option_name in list(current_options_dict.keys()):
        option_info = current_options_dict[option_name]

        if option_info['type'] == 'dict':
            # Recurse into nested dictionary
            if debug:
                 print(f"Entering nested options for '{option_name}'...", file=sys.stderr)
            optimize_options_recursively(option_info['value'], repo_path, root_options_dict, json_options_lookup, forced_options_lookup, debug=debug)
            if debug:
                 print(f"Exiting nested options for '{option_name}'.", file=sys.stderr)
        else:
            # Process non-dict options
            # Check if the option is in the forced list
            # Assumes option_name is globally unique
            if option_name in forced_options_lookup:
                forced_value = forced_options_lookup[option_name]
                print(f"\nSkipping optimization for '{option_name}'. Forcing value to: {forced_value}", file=sys.stderr)
                current_options_dict[option_name]['value'] = forced_value
                continue # Skip optimization for forced options

            # Check if we have possible values for this option from the JSON
            # The check for missing options is done before optimization starts
            if option_name in json_options_lookup and json_options_lookup[option_name]['possible_values']:
                possible_values = json_options_lookup[option_name]['possible_values']
                optimize_option_with_values(current_options_dict, option_name, repo_path, root_options_dict, possible_values, debug=debug)
            elif option_name not in json_options_lookup and option_info['type'] == 'bool':
                 # If the option is not in the JSON lookup but is a boolean, test True/False
                 print(f"\n'{option_name}' not found in JSON, but is boolean. Testing True/False.", file=sys.stderr)
                 optimize_option_with_values(current_options_dict, option_name, repo_path, root_options_dict, [True, False], debug=debug)
            else:
                # If no possible values from JSON and not a boolean, keep the default value
                if debug:
                     print(f"Skipping optimization for '{option_name}' (type: {option_info['type']}). No possible values provided in JSON for optimization and not a boolean.", file=sys.stderr)

