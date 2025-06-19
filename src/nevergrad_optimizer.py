import nevergrad as ng
import sys
import copy
import multiprocessing
from typing import Dict, List, Any, Tuple
import functools # New import

from .base_optimizer import BaseOptimizer
from .data_classes import NevergradConfig, GeneticAlgorithmLookups, WorkerContext
from .clang_format_parser import generate_clang_format_config
from .repo_formatter import run_clang_format_and_count_changes

class NevergradOptimizer(BaseOptimizer):
    """
    Implements the optimization strategy using Nevergrad, a black-box optimization library.
    """

    @staticmethod
    def _nevergrad_objective_function(
        # Nevergrad passes parameters as positional arguments based on instrumentation
        *args,
        # Contextual arguments passed by the NevergradOptimizer.optimize method
        base_options_template: Dict[str, Any],
        lookups: GeneticAlgorithmLookups,
        debug: bool,
        file_sample_percentage: float,
        random_seed: int,
        all_repo_paths: List[str] # List of all temporary repo paths
    ) -> float:
        """
        The objective function to be minimized by Nevergrad.
        It takes Nevergrad's parameters, converts them to a clang-format config,
        runs clang-format, and returns the number of changes.
        This function is designed to be picklable for multiprocessing.
        """
        # Determine the worker's specific repo_path and process_id
        process_id = multiprocessing.current_process()._identity[0] if multiprocessing.current_process()._identity else 0
        # Ensure process_id - 1 is a valid index for all_repo_paths
        # If process_id is 0 (main process), use the first repo_path.
        repo_path_for_worker = all_repo_paths[process_id - 1] if process_id > 0 else all_repo_paths[0]
        worker_context = WorkerContext(repo_path=repo_path_for_worker, process_id=process_id)

        # Reconstruct the flat_options_info from the base template and Nevergrad's parameters
        current_flat_options = copy.deepcopy(base_options_template)

        # Nevergrad's parameters are passed as a single dictionary when using Instrumentation(**kwargs)
        if not args or not isinstance(args[0], dict):
            print(f"Worker {process_id}: Error: Expected Nevergrad parameters as a single dictionary, got {type(args[0])}", file=sys.stderr)
            return float('inf') # Indicate failure

        ng_params = args[0]

        for option_path, ng_value in ng_params.items():
            if option_path in current_flat_options:
                # Ensure type consistency if Nevergrad returns a different type (e.g., float for int)
                target_type = current_flat_options[option_path]['type']
                if target_type == 'int':
                    try:
                        current_flat_options[option_path]['value'] = int(ng_value)
                    except (ValueError, TypeError):
                        if debug:
                            print(f"Worker {process_id}: Warning: Could not convert Nevergrad value '{ng_value}' to int for option '{option_path}'. Skipping.", file=sys.stderr)
                        continue
                elif target_type == 'bool':
                    # Nevergrad's Scalar with integer casting will return 0 or 1.
                    # Convert 0 to False, 1 to True.
                    current_flat_options[option_path]['value'] = bool(ng_value)
                else:
                    current_flat_options[option_path]['value'] = ng_value
            # else: This option was likely not instrumented (e.g., forced, or complex type)
            # and retains its value from base_options_template.

        # Apply forced options (they override any Nevergrad suggestions)
        for forced_path, forced_value in lookups.forced_options_lookup.items():
            if forced_path in current_flat_options:
                current_flat_options[forced_path]['value'] = forced_value

        config_string = generate_clang_format_config(current_flat_options)

        changes = run_clang_format_and_count_changes(
            config_string,
            repo_path=worker_context.repo_path,
            process_id=worker_context.process_id,
            debug=debug,
            file_sample_percentage=file_sample_percentage,
            random_seed=random_seed
        )

        # Nevergrad minimizes, so higher changes mean worse fitness.
        # -1 indicates a git error, which should be treated as very bad.
        # float('inf') is already treated as very bad.
        if changes == -1:
            return float('inf')
        return changes

    def optimize(self,
                 base_options_info: Dict[str, Any],
                 repo_paths: List[str],
                 lookups: GeneticAlgorithmLookups,
                 config: NevergradConfig,
                 file_sample_percentage: float,
                 random_seed: int) -> Dict[str, Any]:
        """
        Optimizes clang-format configuration using Nevergrad.

        Args:
            base_options_info (dict): The initial flat dictionary from clang-format --dump-config.
            repo_paths (list): A list of paths to the temporary git repositories for parallel processing.
            lookups (GeneticAlgorithmLookups): A dataclass containing lookup dictionaries for
                                               option values and forced options.
            config (NevergradConfig): A dataclass containing configuration parameters
                                         for the Nevergrad algorithm.
            file_sample_percentage (float): Percentage of files to sample for fitness calculation.
            random_seed (int): Seed for random file sampling.

        Returns:
            dict: The flat dictionary of the best clang-format configuration found.
        """
        budget = config.budget
        optimizer_name = config.optimizer_name
        num_workers = config.num_workers
        debug = config.debug

        print(f"\nStarting Nevergrad optimization with {optimizer_name}...", file=sys.stderr)
        print(f"Budget: {budget}, Workers: {num_workers}", file=sys.stderr)

        # 1. Build Nevergrad Instrumentation
        # This defines the search space for Nevergrad
        instrumentation_params = {}
        for full_option_path, option_info in base_options_info.items():
            if full_option_path in lookups.forced_options_lookup:
                # Forced options are not part of the search space
                continue

            possible_values = None
            if full_option_path in lookups.json_options_lookup:
                possible_values = lookups.json_options_lookup[full_option_path]['possible_values']

            if option_info['type'] == 'bool':
                # For booleans, use a Scalar that is cast to int (0 or 1)
                # Nevergrad will then pass 0 or 1, which we convert to True/False
                instrumentation_params[full_option_path] = ng.p.Scalar(init=0.0, lower=0.0, upper=1.0).set_integer_casting()
            elif option_info['type'] == 'int':
                if possible_values:
                    # If specific integer values are provided, use Choice
                    # Ensure values are integers, not strings
                    instrumentation_params[full_option_path] = ng.p.Choice([int(v) for v in possible_values])
                else:
                    # For general integers, define a reasonable range.
                    # This is a heuristic; ideally, ranges would be in clang-format-values.json.
                    lower_bound = -100 # Allow negative for offsets
                    upper_bound = 200 # Common max for column limits, indent widths etc.
                    initial_val = option_info['value']
                    if isinstance(initial_val, int):
                        lower_bound = min(lower_bound, initial_val)
                        upper_bound = max(upper_bound, initial_val)
                    
                    instrumentation_params[full_option_path] = ng.p.Scalar(init=initial_val, lower=lower_bound, upper=upper_bound).set_integer_casting()
            elif option_info['type'] == 'str' and possible_values:
                # For string enums, use Choice
                instrumentation_params[full_option_path] = ng.p.Choice(possible_values)
            else:
                # For other types (lists, dicts) or strings without possible values,
                # we fix them to their base value. They are not part of the search space.
                # This means they are implicitly handled by `base_options_template` in the objective function.
                if debug:
                    print(f"Nevergrad: Skipping instrumentation for '{full_option_path}' (type: {option_info['type']}, no possible values or complex type). Will use its base value.", file=sys.stderr)
                pass # Do not add to instrumentation_params

        # Create the instrumentation object
        instrumentation = ng.p.Instrumentation(**instrumentation_params)

        # 2. Create the Nevergrad optimizer
        try:
            # Pass 'instrumentation' as the 'parametrization' argument and 'seed' to the constructor
            optimizer = ng.optimizers.registry[optimizer_name](
                parametrization=instrumentation,
                budget=budget,
                num_workers=num_workers,)
        except KeyError:
            print(f"Error: Nevergrad optimizer '{optimizer_name}' not found. Available optimizers: {list(ng.optimizers.registry.keys())}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error initializing Nevergrad optimizer: {e}", file=sys.stderr)
            sys.exit(1)

        # 3. Run the optimization
        try:
            # Use functools.partial to bind the static context arguments to the objective function
            objective_with_context = functools.partial(
                self._nevergrad_objective_function,
                base_options_template=base_options_info,
                lookups=lookups,
                debug=debug,
                file_sample_percentage=file_sample_percentage,
                random_seed=random_seed,
                all_repo_paths=repo_paths
            )

            recommendation = optimizer.minimize(
                objective_with_context # Pass the partially applied objective function
            )
        except KeyboardInterrupt:
            print("\nCtrl-C detected. Terminating Nevergrad optimization immediately...", file=sys.stderr)
            # Nevergrad handles graceful shutdown on KeyboardInterrupt internally
            # We can still try to get the best recommendation found so far
            recommendation = optimizer.provide_recommendation()
        except Exception as e:
            print(f"An error occurred during Nevergrad optimization: {e}", file=sys.stderr)
            # Attempt to get the best recommendation found so far before exiting
            recommendation = optimizer.provide_recommendation()
            # If no recommendation is available, return the base config
            if recommendation is None:
                print("No recommendation available from Nevergrad. Returning base configuration.", file=sys.stderr)
                return base_options_info
            # Otherwise, proceed to process the partial recommendation
            print("Attempting to process the best recommendation found so far.", file=sys.stderr)


        # 4. Get the best parameters and convert back to clang-format config
        best_ng_params = recommendation.kwargs # Nevergrad returns parameters as kwargs for Instrumentation
        
        # Reconstruct the final optimized config
        optimized_config = copy.deepcopy(base_options_info)
        for option_path, ng_value in best_ng_params.items():
            if option_path in optimized_config:
                target_type = optimized_config[option_path]['type']
                if target_type == 'int':
                    try:
                        optimized_config[option_path]['value'] = int(ng_value)
                    except (ValueError, TypeError):
                        # If conversion fails, keep the original value from base_options_info
                        if debug:
                            print(f"Warning: Failed to convert Nevergrad value '{ng_value}' to int for '{option_path}'. Keeping original value.", file=sys.stderr)
                elif target_type == 'bool':
                    optimized_config[option_path]['value'] = bool(ng_value)
                else:
                    optimized_config[option_path]['value'] = ng_value

        # Apply forced options one last time to the final config
        for forced_path, forced_value in lookups.forced_options_lookup.items():
            if forced_path in optimized_config:
                optimized_config[forced_path]['value'] = forced_value

        print(f"\nNevergrad optimization finished. Best fitness: {recommendation.loss}", file=sys.stderr)
        return optimized_config
