import sys
import copy
import random # New import for genetic algorithm
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


def crossover(parent1_config, parent2_config):
    """
    Performs a uniform crossover between two parent configurations.
    For each option, randomly picks the value from parent1 or parent2.
    """
    child_config = {}
    # Assuming both parents have the same set of options and structure
    for option_path in parent1_config.keys():
        # Ensure we copy the dictionary for the option's info, not just reference it
        if random.random() < 0.5:
            child_config[option_path] = copy.deepcopy(parent1_config[option_path])
        else:
            child_config[option_path] = copy.deepcopy(parent2_config[option_path])
    return child_config

def mutate(individual_config, json_options_lookup, forced_options_lookup, repo_path, debug=False):
    """
    Mutates an individual by selecting one random mutable option and optimizing its value
    by testing all possible values using optimize_option_with_values.
    """
    # Make a deep copy to avoid modifying the original individual directly before evaluation
    mutated_config = copy.deepcopy(individual_config)

    # Find mutable options (not forced, has possible values or is boolean)
    mutable_options = []
    for full_option_path, option_info in mutated_config.items():
        if full_option_path in forced_options_lookup:
            continue # Skip forced options, they are not mutable by the GA

        if (full_option_path in json_options_lookup and json_options_lookup[full_option_path]['possible_values']) or \
           (option_info['type'] == 'bool'):
            mutable_options.append(full_option_path)

    if not mutable_options:
        if debug:
            print("No mutable options found for mutation.", file=sys.stderr)
        return mutated_config # No options to mutate, return original

    # Select a random option to mutate
    option_to_mutate_path = random.choice(mutable_options)
    option_info = mutated_config[option_to_mutate_path]

    possible_values = []
    if option_to_mutate_path in json_options_lookup and json_options_lookup[option_to_mutate_path]['possible_values']:
        possible_values = json_options_lookup[option_to_mutate_path]['possible_values']
    elif option_info['type'] == 'bool':
        possible_values = [True, False]

    if not possible_values:
        # This case should ideally not be reached if mutable_options logic is correct,
        # but as a safeguard, if an option was somehow added to mutable_options without values.
        if debug:
            print(f"Warning: Selected option '{option_to_mutate_path}' for mutation but found no possible values.", file=sys.stderr)
        return mutated_config

    print(f"  Mutating '{option_to_mutate_path}' (current: {option_info['value']})...", file=sys.stderr)
    # Use the existing optimize_option_with_values to find the best value for this single option.
    # It modifies `mutated_config` in place.
    optimize_option_with_values(mutated_config, option_to_mutate_path, repo_path, possible_values, debug=debug)

    return mutated_config

def genetic_optimize_all_options(base_options_info, repo_path, json_options_lookup, forced_options_lookup, num_iterations, max_individuals, debug=False):
    """
    Optimizes clang-format configuration using a genetic algorithm.

    Args:
        base_options_info (dict): The initial flat dictionary from clang-format --dump-config.
        repo_path (str): Path to the git repository.
        json_options_lookup (dict): A dictionary mapping option names to their info
                                    from the JSON file, used to find possible values.
        forced_options_lookup (dict): A dictionary mapping option names to forced values.
        num_iterations (int): Number of generations for the genetic algorithm.
        max_individuals (int): Maximum number of individuals in the population.
        debug (bool): Enable debug output.

    Returns:
        dict: The flat dictionary of the best clang-format configuration found.
    """
    population = []
    print(f"\nInitializing population of {max_individuals} individuals...", file=sys.stderr)

    # Initialize population with copies of the base config and evaluate their fitness
    for i in range(max_individuals):
        individual_config = copy.deepcopy(base_options_info)
        # Apply forced options to initial individuals
        for forced_path, forced_value in forced_options_lookup.items():
            if forced_path in individual_config:
                individual_config[forced_path]['value'] = forced_value

        fitness = run_clang_format_and_count_changes(repo_path, generate_clang_format_config(individual_config), debug=debug)
        population.append({'config': individual_config, 'fitness': fitness})
        print(f"  Individual {i+1}/{max_individuals} initialized with fitness: {fitness}", file=sys.stderr)

    # Find the best individual in the initial population
    best_overall_individual = min(population, key=lambda x: x['fitness'])
    print(f"\nInitial best fitness: {best_overall_individual['fitness']}", file=sys.stderr)

    # Evolution Loop
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---", file=sys.stderr)
        new_generation_candidates = []

        # Elitism: Keep the best individual from the previous generation
        new_generation_candidates.append(best_overall_individual)

        # Generate new individuals through crossover and mutation
        # We generate (max_individuals - 1) new individuals to maintain population size
        # if we are keeping one elite.
        for _ in range(max_individuals - 1):
            # Selection: Simple random selection for parents
            parent1 = random.choice(population)['config']
            parent2 = random.choice(population)['config']

            # Crossover
            child_config = crossover(parent1, parent2)

            # Mutation: Mutate one random option in the child
            mutated_child_config = mutate(child_config, json_options_lookup, forced_options_lookup, repo_path, debug)

            # Apply forced options to the mutated child (ensure they are always respected)
            # This is crucial because crossover might pick a non-forced value for a forced option.
            for forced_path, forced_value in forced_options_lookup.items():
                if forced_path in mutated_child_config:
                    mutated_child_config[forced_path]['value'] = forced_value

            # Evaluate fitness of the mutated child
            child_fitness = run_clang_format_and_count_changes(repo_path, generate_clang_format_config(mutated_child_config), debug=debug)
            new_generation_candidates.append({'config': mutated_child_config, 'fitness': child_fitness})
            print(f"  Generated child with fitness: {child_fitness}", file=sys.stderr)

        # Selection for next generation: Sort all candidates (current population + new children)
        # and take the top `max_individuals` (truncation selection)
        population = sorted(new_generation_candidates, key=lambda x: x['fitness'])[:max_individuals]

        # Update overall best individual
        current_best_in_generation = min(population, key=lambda x: x['fitness'])
        print(f"  Best fitness in current generation: {current_best_in_generation['fitness']}", file=sys.stderr)

        if current_best_in_generation['fitness'] < best_overall_individual['fitness']:
            best_overall_individual = current_best_in_generation
            print(f"  New overall best fitness found: {best_overall_individual['fitness']}", file=sys.stderr)

    print(f"\nGenetic algorithm finished. Best overall fitness: {best_overall_individual['fitness']}", file=sys.stderr)
    return best_overall_individual['config']

