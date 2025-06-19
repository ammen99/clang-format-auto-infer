from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class OptimizationConfig:
    """
    Configuration parameters for the genetic optimization algorithm.
    """
    num_iterations: int
    total_population_size: int
    num_islands: int
    debug: bool
    plot_fitness: bool

@dataclass
class GeneticAlgorithmLookups:
    """
    Lookup data (JSON option values, forced options) needed by the genetic algorithm.
    """
    json_options_lookup: Dict[str, Any]
    forced_options_lookup: Dict[str, Any]

@dataclass
class IslandEvolutionArgs:
    """
    Arguments specific to evolving a single island for one generation.
    These are passed to the worker process for a specific island's task.
    """
    population: List[Dict[str, Any]] # List of {'config': dict, 'fitness': float}
    island_population_size: int
    island_index: int # The logical index of this island (0 to num_islands-1)
    lookups: GeneticAlgorithmLookups # Pass the lookups here
    debug: bool # Pass debug flag here
    file_sample_percentage: float # Percentage of files to sample for fitness calculation
    random_seed: int # Seed for random file sampling

@dataclass
class WorkerContext:
    """
    Context specific to the worker process executing a task.
    This includes the unique temporary repository path and the worker's process ID.
    """
    repo_path: str # Specific repo path for this worker process
    process_id: int # Unique identifier for the worker process (e.g., from multiprocessing.current_process()._identity[0])
