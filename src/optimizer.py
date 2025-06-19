import sys
import copy
import random
import multiprocessing
from typing import List, Tuple, Dict, Any

# Import formatter and config generator
from .repo_formatter import run_clang_format_and_count_changes
from .clang_format_parser import generate_clang_format_config
from .data_classes import OptimizationConfig, GeneticAlgorithmLookups, IslandEvolutionArgs, WorkerContext
from .base_optimizer import BaseOptimizer # New import

# Initialize plt to None to prevent UnboundLocalError warnings from static analyzers
plt = None
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not found. Fitness plotting will be disabled. Install with 'pip install matplotlib' to enable.", file=sys.stderr)
    MATPLOTLIB_AVAILABLE = False


def optimize_option_with_values(flat_options_info: Dict[str, Any], full_option_path: str, possible_values: List[Any], island_args: IslandEvolutionArgs, worker_context: WorkerContext) -> float:
    """
    Optimizes a single option by testing each value in the provided list.
    If multiple values yield the same minimum changes, one is randomly selected.
    Modifies `flat_options_info` in place with the best found value.

    Args:
        flat_options_info (dict): The flat dictionary containing all options.
                                  This dictionary is modified in place.
        full_option_path (str): The dot-separated full name of the option.
        possible_values (list): A list of values to test for this option.
        island_args (IslandEvolutionArgs): Dataclass containing common arguments for the island.
        worker_context (WorkerContext): Dataclass containing worker-specific arguments like
                                        repo_path and process_id.

    Returns:
        float: The minimum number of changes (fitness) achieved by the best value for this option.
               Returns float('inf') if no valid configuration could be found for any tested value.
    """
    option_info = flat_options_info[full_option_path]
    original_value = option_info['value'] # Store original value

    if island_args.debug:
        print(f"Island {island_args.island_index} (Worker {worker_context.process_id}): Optimizing '{full_option_path}' (current: {original_value})...", file=sys.stderr)
        print(f"Island {island_args.island_index} (Worker {worker_context.process_id}):   Testing values: {possible_values}", file=sys.stderr)

    min_changes = float('inf')
    best_values_candidates = [] # Store all values that achieve min_changes

    # Store the original config state to revert if no better option is found
    original_config_state = copy.deepcopy(flat_options_info)

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
                 print(f"Island {island_args.island_index} (Worker {worker_context.process_id}): Warning: Could not convert value '{value_to_test}' to int for option '{full_option_path}'. Skipping.", file=sys.stderr)
                 continue # Skip this value

        # Add other type conversions if necessary (e.g., float, list, etc.)
        # For now, assume other types (like strings for enums) can be used directly


        flat_options_info[full_option_path]['value'] = value_to_test
        config_string = generate_clang_format_config(flat_options_info) # Pass the flat dict

        # run_clang_format_and_count_changes will now exit on critical clang-format error,
        # return float('inf') on invalid config error, or return >= 0 on success, or -1 on git error.
        changes = run_clang_format_and_count_changes(
            config_string,
            repo_path=worker_context.repo_path,
            process_id=worker_context.process_id,
            debug=island_args.debug,
            file_sample_percentage=island_args.file_sample_percentage,
            random_seed=island_args.random_seed
        )

        # We now consider float('inf') as a valid (but high) result, not an error to skip
        if changes != -1: # Only skip if it's a git-related error (-1)
            # Treat float('inf') as a very high change count
            print(f"Island {island_args.island_index} (Worker {worker_context.process_id}):   Testing '{full_option_path}'='{value_to_test}' -> Changes: {changes}", file=sys.stderr)
            if changes < min_changes:
                min_changes = changes
                best_values_candidates = [value_to_test] # New best found, reset list
            elif changes == min_changes:
                best_values_candidates.append(value_to_test) # Another value with same best changes
        else:
            # An error occurred in run_clang_format_and_count_changes (e.g., git diff failed)
            # The error message is already printed by that function.
            # We just need to skip this value and continue with the next one.
            pass # Error message already printed, continue loop


    # --- Decide Best Value ---
    if not best_values_candidates:
        # This happens if all tested values resulted in either a git error (-1) or an invalid clang-format config (float('inf')).
        # In this case, we keep the original value as we couldn't find a better valid one.
        print(f"Island {island_args.island_index} (Worker {worker_context.process_id}): All tests failed or resulted in invalid configurations for '{full_option_path}'. Keeping original value: {original_value}", file=sys.stderr)
        # Restore the original config state if no valid candidate was found
        flat_options_info.update(original_config_state)
        return float('inf') # Indicate that no valid fitness was found for this mutation
    else:
        # Randomly select one of the best values
        best_value = random.choice(best_values_candidates)
        print(f"Island {island_args.island_index} (Worker {worker_context.process_id}): Best value for '{full_option_path}': {best_value} (changes: {min_changes})", file=sys.stderr)
        flat_options_info[full_option_path]['value'] = best_value
        return min_changes


def crossover(parent1_config: Dict[str, Any], parent2_config: Dict[str, Any]) -> Dict[str, Any]:
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

def mutate(individual_config: Dict[str, Any], island_args: IslandEvolutionArgs, worker_context: WorkerContext) -> Tuple[Dict[str, Any], float]:
    """
    Mutates an individual by selecting one random mutable option and optimizing its value
    by testing all possible values using optimize_option_with_values.

    Args:
        individual_config (dict): The configuration of the individual to mutate.
        island_args (IslandEvolutionArgs): Dataclass containing common arguments for the island.
        worker_context (WorkerContext): Dataclass containing worker-specific arguments like
                                        repo_path and process_id.

    Returns:
        tuple[dict, float]: A tuple containing the mutated configuration and its fitness.
    """
    # Make a deep copy to avoid modifying the original individual directly before evaluation
    mutated_config = copy.deepcopy(individual_config)

    # Apply forced options to the mutated child (ensure they are always respected)
    # This ensures that the fitness returned by optimize_option_with_values is accurate
    # for a configuration that includes all forced options.
    for forced_path, forced_value in island_args.lookups.forced_options_lookup.items():
        if forced_path in mutated_config:
            mutated_config[forced_path]['value'] = forced_value

    # Find mutable options (not forced, has possible values or is boolean)
    mutable_options = []
    for full_option_path, option_info in mutated_config.items():
        if full_option_path in island_args.lookups.forced_options_lookup:
            continue # Skip forced options, they are not mutable by the GA

        if (full_option_path in island_args.lookups.json_options_lookup and island_args.lookups.json_options_lookup[full_option_path]['possible_values']) or \
           (option_info['type'] == 'bool'):
            mutable_options.append(full_option_path)

    if not mutable_options:
        if island_args.debug:
            print(f"Island {island_args.island_index} (Worker {worker_context.process_id}): No mutable options found for mutation. Returning original config and its fitness.", file=sys.stderr)
        # If no mutable options, return the original config and a very high fitness
        # to indicate no beneficial mutation occurred.
        return mutated_config, float('inf')

    # Select a random option to mutate
    option_to_mutate_path = random.choice(mutable_options)
    option_info = mutated_config[option_to_mutate_path]

    possible_values = []
    if option_to_mutate_path in island_args.lookups.json_options_lookup and island_args.lookups.json_options_lookup[option_to_mutate_path]['possible_values']:
        possible_values = island_args.lookups.json_options_lookup[option_to_mutate_path]['possible_values']
    elif option_info['type'] == 'bool':
        possible_values = [True, False]

    if not possible_values:
        # This case should ideally not be reached if mutable_options logic is correct,
        # but as a safeguard, if an option was somehow added to mutable_options without values.
        if island_args.debug:
            print(f"Island {island_args.island_index} (Worker {worker_context.process_id}): Warning: Selected option '{option_to_mutate_path}' for mutation but found no possible values. Returning original config and inf fitness.", file=sys.stderr)
        return mutated_config, float('inf') # No values to test, effectively no mutation

    if island_args.debug:
        print(f"Island {island_args.island_index} (Worker {worker_context.process_id}): Mutating '{option_to_mutate_path}' (current: {option_info['value']})...", file=sys.stderr)
    # Use the existing optimize_option_with_values to find the best value for this single option.
    # It modifies `mutated_config` in place and returns the fitness of the resulting config.
    mutated_fitness = optimize_option_with_values(
        mutated_config,
        option_to_mutate_path,
        possible_values,
        island_args, # Pass the IslandEvolutionArgs object
        worker_context # Pass the WorkerContext object
    )

    return mutated_config, mutated_fitness

def _evolve_island_generation_task(island_args: IslandEvolutionArgs, worker_context: WorkerContext) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Helper function to evolve a single island for one generation.
    This function is called by the multiprocessing pool.

    Args:
        island_args (IslandEvolutionArgs): Dataclass containing all arguments
                                           for this island's evolution task.
        worker_context (WorkerContext): Dataclass containing worker-specific arguments.

    Returns:
        tuple: (new_population, best_individual_in_generation)
    """
    # Unpack arguments from the dataclass
    population = island_args.population
    island_population_size = island_args.island_population_size

    new_generation_candidates = []

    # Elitism: Keep the best individual from the current population
    if population:
        best_current_individual = min(population, key=lambda x: x['fitness'])
        new_generation_candidates.append(best_current_individual)
    else:
        # This case should ideally not happen if initial population is correctly created
        return [], {'config': {}, 'fitness': float('inf')}

    # Generate new individuals through crossover and mutation
    num_to_generate = island_population_size - len(new_generation_candidates)
    if num_to_generate < 0:
        num_to_generate = 0

    for _ in range(num_to_generate):
        # Selection: Simple random selection for parents from the current population
        if len(population) < 2:
            parent1 = population[0]['config'] if population else {}
            child_config = copy.deepcopy(parent1)
            if island_args.debug:
                print(f"Island {island_args.island_index} (Worker {worker_context.process_id}): Warning: Island population too small for crossover. Mutating a copy of the best individual.", file=sys.stderr)
        else:
            parent1 = random.choice(population)['config']
            parent2 = random.choice(population)['config']
            child_config = crossover(parent1, parent2)

        # Mutation: Mutate one random option in the child
        # The mutate function now handles applying forced options internally
        mutated_child_config, child_fitness = mutate(
            child_config,
            island_args, # Pass the IslandEvolutionArgs object
            worker_context # Pass the WorkerContext object
        )
        
        new_generation_candidates.append({'config': mutated_child_config, 'fitness': child_fitness})

    # Selection for next generation: Sort all candidates and take the top `island_population_size`
    new_population = sorted(new_generation_candidates, key=lambda x: x['fitness'])[:island_population_size]

    # Find the best individual in this new generation
    best_in_generation = min(new_population, key=lambda x: x['fitness']) if new_population else {'config': {}, 'fitness': float('inf')}

    return new_population, best_in_generation

def _island_evolution_task_wrapper(args_tuple: Tuple[IslandEvolutionArgs, List[str], GeneticAlgorithmLookups, bool, float, int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Wrapper function for multiprocessing.Pool.map.
    It unpacks arguments, determines the worker's specific repo_path and process_id,
    creates a WorkerContext, and then calls the actual island evolution logic.
    """
    island_args_for_this_island, all_repo_paths, lookups, debug, file_sample_percentage, random_seed = args_tuple

    # Get the unique process ID for this worker
    # _identity[0] gives a unique integer for each worker process (starts from 1)
    process_id = multiprocessing.current_process()._identity[0]
    
    # Select the dedicated repo path for this worker based on its process ID
    # Ensure process_id - 1 is a valid index for all_repo_paths
    repo_path_for_worker = all_repo_paths[process_id - 1]

    # Create a WorkerContext for this specific worker process
    worker_context = WorkerContext(repo_path=repo_path_for_worker, process_id=process_id)

    # Update the island_args with the lookups, debug, sampling, and random_seed
    # These were passed separately to the wrapper because they are constant across all islands
    # but need to be part of IslandEvolutionArgs for the inner functions.
    # This is a bit of a hack to avoid modifying IslandEvolutionArgs in place in the main process,
    # but it works for passing context down.
    island_args_for_this_island.lookups = lookups
    island_args_for_this_island.debug = debug
    island_args_for_this_island.file_sample_percentage = file_sample_percentage
    island_args_for_this_island.random_seed = random_seed

    # Now call the actual island evolution function
    return _evolve_island_generation_task(island_args_for_this_island, worker_context)


def _perform_migration(populations: List[List[Dict[str, Any]]], debug: bool = False):
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


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """
    Implements the genetic algorithm optimization strategy with an island model.
    """
    def optimize(self,
                 base_options_info: Dict[str, Any],
                 repo_paths: List[str],
                 lookups: GeneticAlgorithmLookups,
                 ga_config: OptimizationConfig,
                 file_sample_percentage: float,
                 random_seed: int) -> Dict[str, Any]:
        """
        Optimizes clang-format configuration using a genetic algorithm with an island model.

        Args:
            base_options_info (dict): The initial flat dictionary from clang-format --dump-config.
            repo_paths (list): A list of paths to the temporary git repositories for parallel processing.
            lookups (GeneticAlgorithmLookups): A dataclass containing lookup dictionaries for
                                               option values and forced options.
            ga_config (OptimizationConfig): A dataclass containing configuration parameters
                                            for the genetic algorithm.
            file_sample_percentage (float): Percentage of files to sample for fitness calculation.
            random_seed (int): Seed for random file sampling.

        Returns:
            dict: The flat dictionary of the best clang-format configuration found.
        """
        # Unpack from ga_config
        num_iterations = ga_config.num_iterations
        total_population_size = ga_config.total_population_size
        num_islands = ga_config.num_islands
        debug = ga_config.debug
        plot_fitness = ga_config.plot_fitness

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

        # Create the base individual configuration and calculate its fitness once
        base_individual_config = copy.deepcopy(base_options_info)
        for forced_path, forced_value in lookups.forced_options_lookup.items():
            if forced_path in base_individual_config:
                base_individual_config[forced_path]['value'] = forced_value

        print("Calculating initial base configuration fitness...", file=sys.stderr)
        # Use the first repo path for initial fitness calculation, as all copies are identical
        initial_repo_path = repo_paths[0] if repo_paths else None
        if initial_repo_path is None:
            print("Error: No repository paths provided for initialization.", file=sys.stderr)
            return {} # Or raise an error

        # Create a dummy WorkerContext for the initial fitness calculation (main process acts as worker 0)
        initial_worker_context = WorkerContext(
            repo_path=initial_repo_path,
            process_id=0 # Main process can be considered worker 0 for this initial calculation
        )

        base_fitness = run_clang_format_and_count_changes(
            generate_clang_format_config(base_individual_config),
            repo_path=initial_worker_context.repo_path,
            process_id=initial_worker_context.process_id,
            debug=debug,
            file_sample_percentage=file_sample_percentage,
            random_seed=random_seed
        )
        print(f"Initial base configuration fitness: {base_fitness}", file=sys.stderr)

        # Initialize each island's population with copies of the base individual
        for i in range(num_islands):
            island_pop = []
            for _ in range(island_population_size):
                # All initial individuals are identical to the base config
                individual_config = copy.deepcopy(base_individual_config)
                island_pop.append({'config': individual_config, 'fitness': base_fitness})
                # No need to print fitness for each, as it's the same
            populations.append(island_pop)

        # Find the best individual in the initial overall population (which is just the base_fitness)
        best_overall_individual = {'config': base_individual_config, 'fitness': base_fitness}
        print(f"\nInitial overall best fitness: {best_overall_individual['fitness']}", file=sys.stderr)

        # Data structure to store best fitness for each island over time
        fitness_history_per_island = [[] for _ in range(num_islands)]
        # Populate initial fitness history for plotting
        for i in range(num_islands):
            fitness_history_per_island[i].append(base_fitness)


        # Initialize plot variables to None/empty list
        global plt # Access the global plt
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
        MIGRATION_INTERVAL = 15

        # Create the multiprocessing pool
        # The number of processes in the pool is limited by the number of available repo copies
        num_processes = len(repo_paths)
        pool = multiprocessing.Pool(processes=num_processes)

        interrupted = False # Flag to indicate if optimization was interrupted by Ctrl-C

        try:
            # Evolution Loop
            for iteration in range(num_iterations):
                print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---", file=sys.stderr)

                # Prepare arguments for each island's evolution task
                tasks_args = []
                for i, island_pop in enumerate(populations):
                    # Create IslandEvolutionArgs for this specific island
                    island_args_for_this_island = IslandEvolutionArgs(
                        population=island_pop,
                        island_population_size=island_population_size,
                        island_index=i, # Pass the logical island index
                        lookups=lookups, # These will be passed to the wrapper and then assigned
                        debug=debug,     # to the IslandEvolutionArgs object inside the worker
                        file_sample_percentage=file_sample_percentage,
                        random_seed=random_seed
                    )
                    # The wrapper will handle assigning the correct repo_path and process_id
                    tasks_args.append((island_args_for_this_island, repo_paths, lookups, debug, file_sample_percentage, random_seed))

                # Run island evolutions in parallel
                results = pool.map(_island_evolution_task_wrapper, tasks_args)

                # Process results from parallel evolution
                for i, (new_island_pop, best_in_island_for_this_gen) in enumerate(results):
                    populations[i] = new_island_pop # Update the island's population
                    print(f"    Island {i + 1} best fitness: {best_in_island_for_this_gen['fitness']}", file=sys.stderr)

                    # Store fitness for plotting
                    fitness_history_per_island[i].append(best_in_island_for_this_gen['fitness'])

                    # Update overall best individual if this island found a better one
                    if best_in_island_for_this_gen['fitness'] < best_overall_individual['fitness']:
                        best_overall_individual = best_in_island_for_this_gen
                        print(f"    New overall best fitness found: {best_overall_individual['fitness']}", file=sys.stderr)

                # Update plot after all islands have evolved in this iteration
                if plot_fitness and not interrupted: # Only update if not interrupted
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
                if num_islands > 1 and (iteration + 1) % MIGRATION_INTERVAL == 0 and not interrupted:
                    print(f"\n--- Performing migration at iteration {iteration + 1} ---", file=sys.stderr)
                    _perform_migration(populations, debug)

                    # After migration, re-find the overall best individual from the updated populations
                    all_individuals_after_migration = [ind for island_pop in populations for ind in island_pop]
                    if all_individuals_after_migration:
                        current_overall_best_after_migration = min(all_individuals_after_migration, key=lambda x: x['fitness'])
                        if current_overall_best_after_migration['fitness'] < best_overall_individual['fitness']:
                            best_overall_individual = current_overall_best_after_migration
                            print(f"  New overall best fitness found after migration: {current_overall_best_after_migration['fitness']}", file=sys.stderr)
                        else:
                            print(f"  Overall best fitness remains: {best_overall_individual['fitness']} after migration.", file=sys.stderr)
                    else:
                        print("Warning: All populations are empty after migration. This should not happen.", file=sys.stderr)

        except KeyboardInterrupt:
            print("\nCtrl-C detected. Terminating optimization immediately...", file=sys.stderr)
            interrupted = True
            # Close matplotlib plots if they are open
            if MATPLOTLIB_AVAILABLE and plt:
                try:
                    plt.close('all')
                    print("Matplotlib plots closed.", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Could not close matplotlib plots: {e}", file=sys.stderr)

        finally:
            if interrupted:
                print("Forcing termination of worker pool...", file=sys.stderr)
                pool.terminate() # Terminate workers immediately
            else:
                pool.close() # Allow workers to finish current tasks gracefully

            pool.join() # Wait for all workers to exit (either gracefully or terminated)
            print("Worker pool shut down.", file=sys.stderr)

        print(f"\nGenetic algorithm finished. Best overall fitness: {best_overall_individual['fitness']}", file=sys.stderr)

        # Keep the plot open at the end if it was generated AND optimization was not interrupted
        if plot_fitness and MATPLOTLIB_AVAILABLE and not interrupted:
            assert plt is not None
            plt.ioff()
            plt.show()

        return best_overall_individual['config']
