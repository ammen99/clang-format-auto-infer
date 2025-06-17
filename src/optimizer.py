import sys
import copy
import random
import signal

# Import formatter and config generator
from .repo_formatter import run_clang_format_and_count_changes
from .clang_format_parser import generate_clang_format_config

# Initialize plt to None to prevent UnboundLocalError warnings from static analyzers
plt = None
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not found. Fitness plotting will be disabled. Install with 'pip install matplotlib' to enable.", file=sys.stderr)
    MATPLOTLIB_AVAILABLE = False

# Global flag to signal early termination
_stop_optimization_flag = False

def _signal_handler(sig, frame):
    """
    Signal handler for SIGINT (Ctrl-C).
    Sets a global flag to stop the optimization loop gracefully.
    """
    global _stop_optimization_flag
    del sig, frame
    print("\nCtrl-C detected. Stopping optimization gracefully...", file=sys.stderr)
    _stop_optimization_flag = True

# Register the signal handler when the module is loaded
signal.signal(signal.SIGINT, _signal_handler)


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
        # Check for stop flag before each test
        global _stop_optimization_flag
        if _stop_optimization_flag:
            print(f"  Optimization interrupted for '{full_option_path}'. Keeping original value.", file=sys.stderr)
            flat_options_info[full_option_path]['value'] = original_value # Restore original if interrupted
            return # Exit early from this function

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

def _evolve_island_generation(population, repo_path, json_options_lookup, forced_options_lookup, island_population_size, debug=False):
    """
    Performs one generation of evolution for a single island's population.

    Args:
        population (list): The current list of individuals for this island.
                           Each individual is {'config': dict, 'fitness': float}.
        repo_path (str): Path to the git repository.
        json_options_lookup (dict): Lookup for possible option values.
        forced_options_lookup (dict): Lookup for forced option values.
        island_population_size (int): The target size of this island's population.
        debug (bool): Enable debug output.

    Returns:
        tuple: (new_population, best_individual_in_generation)
    """
    new_generation_candidates = []

    # Elitism: Keep the best individual from the current population
    if population:
        best_current_individual = min(population, key=lambda x: x['fitness'])
        new_generation_candidates.append(best_current_individual)
    else:
        # This case should ideally not happen if initial population is correctly created
        # but as a safeguard, if population is empty, we can't evolve.
        return [], {'config': {}, 'fitness': float('inf')}

    # Generate new individuals through crossover and mutation
    # We generate (island_population_size - 1) new individuals to maintain population size
    # if we are keeping one elite.
    num_to_generate = island_population_size - len(new_generation_candidates)
    if num_to_generate < 0:
        num_to_generate = 0 # Should not happen if island_population_size >= 1

    for _ in range(num_to_generate):
        # Check for stop flag before generating new individual
        global _stop_optimization_flag
        if _stop_optimization_flag:
            print("  Evolution interrupted for current island.", file=sys.stderr)
            break # Break from this inner loop

        # Selection: Simple random selection for parents from the current population
        # Ensure there are at least two individuals for crossover
        if len(population) < 2:
            # If population is too small for crossover, just mutate the best existing one
            # This case should be prevented by minimum island_population_size, but as a safeguard
            parent1 = population[0]['config'] if population else {} # Fallback to empty if population is somehow empty
            child_config = copy.deepcopy(parent1)
            if debug:
                print("Warning: Island population too small for crossover. Mutating a copy of the best individual.", file=sys.stderr)
        else:
            parent1 = random.choice(population)['config']
            parent2 = random.choice(population)['config']
            child_config = crossover(parent1, parent2)

        # Mutation: Mutate one random option in the child
        mutated_child_config = mutate(child_config, json_options_lookup, forced_options_lookup, repo_path, debug)

        # Apply forced options to the mutated child (ensure they are always respected)
        for forced_path, forced_value in forced_options_lookup.items():
            if forced_path in mutated_child_config:
                mutated_child_config[forced_path]['value'] = forced_value

        # Evaluate fitness of the mutated child
        child_fitness = run_clang_format_and_count_changes(repo_path, generate_clang_format_config(mutated_child_config), debug=debug)
        new_generation_candidates.append({'config': mutated_child_config, 'fitness': child_fitness})

    # Selection for next generation: Sort all candidates and take the top `island_population_size`
    new_population = sorted(new_generation_candidates, key=lambda x: x['fitness'])[:island_population_size]

    # Find the best individual in this new generation
    best_in_generation = min(new_population, key=lambda x: x['fitness']) if new_population else {'config': {}, 'fitness': float('inf')}

    return new_population, best_in_generation

def _perform_migration(populations, debug=False):
    """
    Performs migration between islands.
    Each island sends its best individual to a randomly chosen other island,
    replacing a random individual in the target island.
    """
    if len(populations) < 2:
        if debug:
            print("Skipping migration: Less than 2 islands.", file=sys.stderr)
        return

    migrants = []
    # Collect the best individual from each island
    for i, island_pop in enumerate(populations):
        if island_pop:
            best_individual = min(island_pop, key=lambda x: x['fitness'])
            migrants.append({'source_island_idx': i, 'individual': best_individual})
        else:
            if debug:
                print(f"Warning: Island {i} is empty, cannot select migrant.", file=sys.stderr)

    if debug:
        print(f"Performing migration with {len(migrants)} migrants.", file=sys.stderr)

    # Distribute migrants
    for migrant_info in migrants:
        source_idx = migrant_info['source_island_idx']
        migrant = migrant_info['individual']

        # Choose a random target island different from the source
        target_island_idx = source_idx
        # Ensure there's at least one other island to migrate to
        if len(populations) > 1:
            while target_island_idx == source_idx:
                target_island_idx = random.randrange(len(populations))
        else:
            # If only one island, no migration possible
            continue

        target_population = populations[target_island_idx]

        if not target_population:
            # If target island is empty, just add the migrant
            target_population.append(migrant)
            if debug:
                print(f"  Migrant from island {source_idx} added to empty island {target_island_idx}.", file=sys.stderr)
        else:
            # Replace a random individual in the target island with the migrant
            # Ensure target_population has at least one element before random.choice
            if len(target_population) > 0:
                individual_to_replace = random.choice(target_population)
                target_population.remove(individual_to_replace)
                target_population.append(migrant)
                if debug:
                    print(f"  Migrant from island {source_idx} replaced an individual in island {target_island_idx}.", file=sys.stderr)
            else:
                # This case should be rare if populations are maintained, but as a safeguard
                target_population.append(migrant)
                if debug:
                    print(f"  Migrant from island {source_idx} added to island {target_island_idx} (was unexpectedly empty).", file=sys.stderr)


def genetic_optimize_all_options(base_options_info, repo_path, json_options_lookup, forced_options_lookup, num_iterations, total_population_size, num_islands, debug=False, plot_fitness=False):
    """
    Optimizes clang-format configuration using a genetic algorithm with an island model.

    Args:
        base_options_info (dict): The initial flat dictionary from clang-format --dump-config.
        repo_path (str): Path to the git repository.
        json_options_lookup (dict): A dictionary mapping option names to their info
                                    from the JSON file, used to find possible values.
        forced_options_lookup (dict): A dictionary mapping option names to forced values.
        num_iterations (int): Number of generations for the genetic algorithm.
        total_population_size (int): The total number of individuals across all islands.
        num_islands (int): The number of independent populations (islands).
        debug (bool): Enable debug output.
        plot_fitness (bool): If True, visualize the best fitness over time for each island.

    Returns:
        dict: The flat dictionary of the best clang-format configuration found.
    """
    global _stop_optimization_flag
    _stop_optimization_flag = False # Reset flag for a new run

    if num_islands < 1:
        print("Error: Number of islands must be at least 1. Setting to 1.", file=sys.stderr)
        num_islands = 1

    # Determine population size per island
    # Ensure a minimum of 5 individuals per island for meaningful evolution
    MIN_INDIVIDUALS_PER_ISLAND = 5
    island_population_size = max(MIN_INDIVIDUALS_PER_ISLAND, total_population_size // num_islands)

    # Adjust total_population_size if the minimum per island makes it larger
    if island_population_size * num_islands > total_population_size:
        total_population_size = island_population_size * num_islands
        print(f"Adjusted total population size to {total_population_size} to ensure at least {island_population_size} individuals per island.", file=sys.stderr)
    elif total_population_size < num_islands * MIN_INDIVIDUALS_PER_ISLAND:
        print(f"Warning: Total population size ({total_population_size}) is too small for {num_islands} islands "
              f"with a minimum of {MIN_INDIVIDUALS_PER_ISLAND} individuals per island. "
              f"Each island will have {island_population_size} individuals.", file=sys.stderr)


    populations = []
    print(f"\nInitializing {num_islands} islands, each with {island_population_size} individuals (total: {total_population_size})...", file=sys.stderr)

    # Initialize each island's population
    for i in range(num_islands):
        island_pop = []
        for j in range(island_population_size):
            if _stop_optimization_flag: # Check flag during initialization too
                print("Initialization interrupted.", file=sys.stderr)
                break
            individual_config = copy.deepcopy(base_options_info)
            # Apply forced options to initial individuals
            for forced_path, forced_value in forced_options_lookup.items():
                if forced_path in individual_config:
                    individual_config[forced_path]['value'] = forced_value

            fitness = run_clang_format_and_count_changes(repo_path, generate_clang_format_config(individual_config), debug=debug)
            island_pop.append({'config': individual_config, 'fitness': fitness})
            print(f"  Island {i+1}, Individual {j+1}/{island_population_size} initialized with fitness: {fitness}", file=sys.stderr)
        populations.append(island_pop)
        if _stop_optimization_flag:
            break # Break outer loop if initialization was interrupted

    # Find the best individual in the initial overall population
    all_individuals = [ind for island_pop in populations for ind in island_pop]
    best_overall_individual = min(all_individuals, key=lambda x: x['fitness']) if all_individuals else {'config': {}, 'fitness': float('inf')}
    print(f"\nInitial overall best fitness: {best_overall_individual['fitness']}", file=sys.stderr)

    # Data structure to store best fitness for each island over time
    fitness_history_per_island = [[] for _ in range(num_islands)]

    # Initialize plot variables to None/empty list
    fig = None
    ax = None
    lines = []

    # Setup plot if requested and matplotlib is available
    if plot_fitness and MATPLOTLIB_AVAILABLE:
        assert plt is not None # Assert plt is not None here
        plt.ion() # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Best Fitness Over Generations for Each Island")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness (Number of Changes)")
        ax.grid(True)
        lines = [ax.plot([], [], label=f'Island {i+1}')[0] for i in range(num_islands)]
        ax.legend()
        fig.show() # Show the figure immediately
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01) # Give time for plot to render
    elif plot_fitness and not MATPLOTLIB_AVAILABLE:
        print("Plotting requested but matplotlib is not available. Skipping plot.", file=sys.stderr)
        plot_fitness = False # Disable plotting for the rest of the function


    # Migration interval (e.g., migrate every 10 generations)
    MIGRATION_INTERVAL = 25

    # Evolution Loop
    for iteration in range(num_iterations):
        if _stop_optimization_flag:
            print("\nOptimization loop interrupted by user.", file=sys.stderr)
            break # Exit the main optimization loop

        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---", file=sys.stderr)

        # Evolve each island independently
        for i, island_pop in enumerate(populations):
            if _stop_optimization_flag: # Check flag before evolving next island
                print(f"  Skipping remaining islands due to interruption.", file=sys.stderr)
                break
            print(f"  Evolving Island {i + 1}...", file=sys.stderr)
            new_island_pop, best_in_island_for_this_gen = _evolve_island_generation(
                island_pop, repo_path, json_options_lookup, forced_options_lookup, island_population_size, debug
            )
            populations[i] = new_island_pop # Update the island's population
            print(f"    Island {i + 1} best fitness: {best_in_island_for_this_gen['fitness']}", file=sys.stderr)

            # Store fitness for plotting
            fitness_history_per_island[i].append(best_in_island_for_this_gen['fitness'])

            # Update overall best individual if this island found a better one
            if best_in_island_for_this_gen['fitness'] < best_overall_individual['fitness']:
                best_overall_individual = best_in_island_for_this_gen
                print(f"    New overall best fitness found: {best_overall_individual['fitness']}", file=sys.stderr)

        # Update plot after all islands have evolved in this iteration
        if plot_fitness and not _stop_optimization_flag: # Only update if not interrupted
            assert ax is not None
            assert fig is not None
            assert plt is not None
            for i, history in enumerate(fitness_history_per_island):
                lines[i].set_data(range(len(history)), history)
            ax.relim() # Recalculate limits
            ax.autoscale_view() # Autoscale axes
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01) # Short pause to allow plot to update

        # Perform migration periodically if there's more than one island
        if num_islands > 1 and (iteration + 1) % MIGRATION_INTERVAL == 0 and not _stop_optimization_flag:
            print(f"\n--- Performing migration at iteration {iteration + 1} ---", file=sys.stderr)
            _perform_migration(populations, debug)

            # After migration, re-find the overall best individual from the updated populations
            all_individuals_after_migration = [ind for island_pop in populations for ind in island_pop]
            if all_individuals_after_migration:
                current_overall_best_after_migration = min(all_individuals_after_migration, key=lambda x: x['fitness'])
                if current_overall_best_after_migration['fitness'] < best_overall_individual['fitness']:
                    best_overall_individual = current_overall_best_after_migration
                    print(f"  New overall best fitness found after migration: {best_overall_individual['fitness']}", file=sys.stderr)
                else:
                    print(f"  Overall best fitness remains: {best_overall_individual['fitness']} after migration.", file=sys.stderr)
            else:
                print("Warning: All populations are empty after migration. This should not happen.", file=sys.stderr)


    print(f"\nGenetic algorithm finished. Best overall fitness: {best_overall_individual['fitness']}", file=sys.stderr)

    # Keep the plot open at the end if it was generated
    if plot_fitness and MATPLOTLIB_AVAILABLE: # Check MATPLOTLIB_AVAILABLE again before final show
        assert plt is not None # Assert plt is not None here
        plt.ioff() # Turn off interactive mode
        plt.show() # Show the final plot and block until closed

    return best_overall_individual['config']
