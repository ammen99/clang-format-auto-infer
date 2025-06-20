import nevergrad as ng
import sys
import copy
import multiprocessing
from typing import Dict, List, Any
import functools
import concurrent.futures

from .base_optimizer import BaseOptimizer
from .data_classes import NevergradConfig, GeneticAlgorithmLookups, WorkerContext
from .clang_format_parser import generate_clang_format_config
from .repo_formatter import run_clang_format_and_count_changes

# Initialize plt to None to prevent UnboundLocalError warnings from static analyzers
plt = None
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not found. Fitness plotting will be disabled. Install with 'pip install matplotlib' to enable.", file=sys.stderr)


class NevergradOptimizer(BaseOptimizer):
    """
    Implements the optimization strategy using Nevergrad, a black-box optimization library.
    """

    def __init__(self, config: NevergradConfig):
        self.config: NevergradConfig;
        super().__init__(config) # Call parent constructor and store config

    @staticmethod
    def _nevergrad_objective_function(
        # Contextual arguments passed by functools.partial
        base_options_template: Dict[str, Any],
        lookups: GeneticAlgorithmLookups,
        debug: bool,
        file_sample_percentage: float,
        random_seed: int,
        all_repo_paths: List[str],
        # Nevergrad passes parameters as keyword arguments based on instrumentation
        # These are the parameters from the search space (e.g., AlignAfterOpenBracket=value)
        **ng_params, # This must be the last argument
    ) -> float:
        """
        The objective function to be minimized by Nevergrad.
        It takes Nevergrad's parameters, converts them to a clang-format config,
        runs clang-format, and returns the number of changes.
        This function is designed to be picklable for multiprocessing.
        """
        # Determine the worker's specific repo_path and process_id
        # The process_id here is primarily for logging and identifying the OS process.
        # Nevergrad's ProcessPoolExecutor assigns unique _identity[0] to each worker.
        process_id = multiprocessing.current_process()._identity[0] if multiprocessing.current_process()._identity else 0

        # Select the dedicated repo path for this worker based on its process ID
        # Nevergrad's workers typically start with _identity[0] = 1, 2, ...
        # So, process_id - 1 maps to the correct index in all_repo_paths.
        # If process_id is 0 (e.g., main process for a single worker setup or direct call), use index 0.
        repo_path_for_worker = all_repo_paths[process_id - 1] if process_id > 0 else all_repo_paths[0]

        if debug:
            print(f"Worker {process_id}: Using repo path: {repo_path_for_worker}", file=sys.stderr)

        worker_context = WorkerContext(repo_path=repo_path_for_worker, process_id=process_id)

        # Reconstruct the flat_options_info from the base template and Nevergrad's parameters
        current_flat_options = copy.deepcopy(base_options_template)

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
                 file_sample_percentage: float,
                 random_seed: int,
                 ) -> Dict[str, Any]:
        """
        Optimizes clang-format configuration using Nevergrad.

        Args:
            base_options_info (dict): The initial flat dictionary from clang-format --dump-config.
            repo_paths (list): A list of paths to the temporary git repositories for parallel processing.
            lookups (GeneticAlgorithmLookups): A dataclass containing lookup dictionaries for
                                               option values and forced options.
            file_sample_percentage (float): Percentage of files to sample for fitness calculation.
            random_seed (int): Seed for random file sampling.

        Returns:
            dict: The flat dictionary of the best clang-format configuration found.
        """
        # Access config from self.config
        budget = self.config.budget
        optimizer_name = self.config.optimizer_name
        num_workers = self.config.num_workers
        debug = self.config.debug
        plot_fitness = self.config.plot_fitness

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
            elif option_info['type'] == 'int' and possible_values:
                instrumentation_params[full_option_path] = ng.p.Choice([int(v) for v in possible_values])
            elif option_info['type'] == 'str' and possible_values:
                # For string enums, use Choice
                instrumentation_params[full_option_path] = ng.p.Choice(possible_values)
            else:
                # For other types (lists, dicts) or strings without possible values,
                # we fix them to their base value. They are not part of the search space.
                # This means they are implicitly handled by `base_options_template` in the objective function.
                print(f"Nevergrad: Skipping instrumentation for '{full_option_path}' (type: {option_info['type']}, no possible values or complex type). Will use its base value.", file=sys.stderr)

        # Create the instrumentation object
        instrumentation = ng.p.Instrumentation(**instrumentation_params)

        # 2. Create the Nevergrad optimizer
        try:
            # Pass 'instrumentation' as the 'parametrization' argument.
            # Removed 'seed=random_seed' as it's not universally supported by all optimizers.
            optimizer = ng.optimizers.registry[optimizer_name](
                parametrization=instrumentation,
                budget=budget,
                num_workers=num_workers,
            )
        except KeyError:
            print(f"Error: Nevergrad optimizer '{optimizer_name}' not found. Available optimizers: {list(ng.optimizers.registry.keys())}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error initializing Nevergrad optimizer: {e}", file=sys.stderr)
            sys.exit(1)

        # 3. Run the optimization using ask/tell loop
        best_fitness_history = [] # To store best fitness over evaluations
        fig = None
        ax = None
        line = None
        interrupted = False # Flag for KeyboardInterrupt

        # Setup plot if requested and matplotlib is available
        if plot_fitness and MATPLOTLIB_AVAILABLE:
            assert plt is not None # Assert plt is not None here
            plt.ion() # Turn on interactive mode
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"Nevergrad Optimization ({optimizer_name}) - Best Fitness Over Evaluations")
            ax.set_xlabel("Evaluations")
            ax.set_ylabel("Best Fitness (Number of Changes)")
            ax.grid(True)
            line, = ax.plot([], [], label="Best Fitness") # Comma is important for unpacking
            ax.legend()
            fig.show() # Show the figure immediately
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01) # Give time for plot to render
        elif plot_fitness and not MATPLOTLIB_AVAILABLE:
            print("Plotting requested but matplotlib is not available. Skipping plot.", file=sys.stderr)
            plot_fitness = False # Disable plotting for the rest of the function

        executor = None # Initialize executor to None
        pending_futures: Dict[concurrent.futures.Future, ng.p.Parameter] = {}
        current_eval_count = 0

        try:
            # Use functools.partial to bind the static context arguments to the objective function
            objective_with_context = functools.partial(
                self._nevergrad_objective_function,
                base_options_template=base_options_info,
                lookups=lookups,
                debug=debug,
                file_sample_percentage=file_sample_percentage,
                random_seed=random_seed, # random_seed is still passed here for file sampling
                all_repo_paths=repo_paths,
            )

            # Create a ProcessPoolExecutor for Nevergrad to use for parallel evaluations
            # max_workers should be equal to num_workers specified for Nevergrad
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)
            print(f"Nevergrad: Using ProcessPoolExecutor with {num_workers} workers.", file=sys.stderr)

            def _submit_next_evaluation():
                nonlocal current_eval_count
                if current_eval_count < budget:
                    candidate = optimizer.ask()
                    future = executor.submit(objective_with_context, **candidate.kwargs)
                    pending_futures[future] = candidate
                    current_eval_count += 1
                    if debug:
                        print(f"Nevergrad: Submitted new task. Active tasks: {len(pending_futures)}/{num_workers}. Total evaluations: {current_eval_count}/{budget}", file=sys.stderr)
                    return True
                return False

            def _refill_queue():
                while len(pending_futures) < num_workers:
                    if not _submit_next_evaluation():
                        break # Budget reached or no more tasks to submit

            _refill_queue() # initial batch

            # Main ask/tell loop: continue as long as there are pending tasks or budget allows new ones
            while (current_eval_count < budget or pending_futures) and not interrupted:
                if not pending_futures and current_eval_count >= budget:
                    # No more tasks to run and budget is exhausted
                    break

                # Wait for at least one future to complete
                done, _ = concurrent.futures.wait(pending_futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)

                for completed_future in done:
                    candidate = pending_futures.pop(completed_future) # Remove from pending
                    try:
                        loss = completed_future.result()
                        optimizer.tell(candidate, loss)
                        best_loss_so_far = loss if not best_fitness_history else min(loss, min(best_fitness_history))
                        best_fitness_history.append(best_loss_so_far)

                        if debug:
                            print(f"Nevergrad: Evaluation {len(best_fitness_history)} (Loss: {loss}, Best so far: {best_loss_so_far})", file=sys.stderr)

                        if plot_fitness and MATPLOTLIB_AVAILABLE and not interrupted:
                            assert ax is not None
                            assert fig is not None
                            assert plt is not None
                            assert line is not None
                            line.set_data(range(len(best_fitness_history)), best_fitness_history)
                            ax.relim() # Recalculate limits
                            ax.autoscale_view() # Autoscale axes
                            fig.canvas.draw()
                            fig.canvas.flush_events()
                            plt.pause(0.01) # Short pause to allow plot to update

                    except KeyboardInterrupt:
                        print("\nCtrl-C detected during evaluation. Terminating Nevergrad optimization immediately...", file=sys.stderr)
                        interrupted = True
                        break # Break from processing completed futures
                    except Exception as e:
                        print(f"Nevergrad: Error during evaluation: {e}", file=sys.stderr)
                        # Tell Nevergrad about the error with a high loss to penalize it
                        optimizer.tell(candidate, float('inf'))
                        # Continue to the next completed future, don't break the outer loop

                if interrupted:
                    break # Break from main while loop if interrupted
                _refill_queue()

            recommendation = optimizer.provide_recommendation()

        except KeyboardInterrupt:
            print("\nCtrl-C detected. Terminating Nevergrad optimization immediately...", file=sys.stderr)
            interrupted = True
            recommendation = optimizer.provide_recommendation() # Get best found so far
        except Exception as e:
            print(f"An error occurred during Nevergrad optimization: {e}", file=sys.stderr)
            recommendation = optimizer.provide_recommendation()
            if recommendation is None:
                print("No recommendation available from Nevergrad. Returning base configuration.", file=sys.stderr)
                return base_options_info
            print("Attempting to process the best recommendation found so far.", file=sys.stderr)
        finally:
            if executor:
                print("Shutting down Nevergrad's ProcessPoolExecutor...", file=sys.stderr)
                # Cancel remaining futures before shutdown
                for future in pending_futures.keys():
                    future.cancel()
                executor.shutdown(wait=True)
                print("Nevergrad's ProcessPoolExecutor shut down.", file=sys.stderr)

            # Close matplotlib plots if they are open and not already closed by interrupt handler
            if plot_fitness and MATPLOTLIB_AVAILABLE:
                assert plt
                try:
                    if interrupted: # If interrupted, plots might already be closed or in a bad state
                        plt.close('all')
                        print("Matplotlib plots closed due to interruption.", file=sys.stderr)
                    else: # If not interrupted, keep plot open briefly then close
                        plt.ioff() # Turn off interactive mode
                        plt.show() # Show final plot and block until closed
                        print("Matplotlib plot displayed and closed.", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Could not close matplotlib plots: {e}", file=sys.stderr)


        # 4. Get the best parameters and convert back to clang-format config
        if recommendation is None:
            print("Warning: No recommendation provided by Nevergrad. Returning base configuration.", file=sys.stderr)
            return base_options_info

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
