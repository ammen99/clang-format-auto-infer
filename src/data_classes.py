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
    """
    population: List[Dict[str, Any]] # List of {'config': dict, 'fitness': float}
    island_population_size: int
    repo_path: str # Specific repo path for this island's worker
    lookups: GeneticAlgorithmLookups # Pass the lookups here
    debug: bool # Pass debug flag here
    file_sample_percentage: float # New: Percentage of files to sample for fitness calculation
    random_seed: int # New: Seed for random file sampling
    worker_id: int # New: Identifier for the worker process
