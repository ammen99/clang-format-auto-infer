from abc import ABC, abstractmethod
from typing import Dict, List, Any
from .data_classes import OptimizationConfig, GeneticAlgorithmLookups

class BaseOptimizer(ABC):
    """
    Abstract base class for clang-format configuration optimizers.
    Defines the interface for different optimization strategies.
    """

    @abstractmethod
    def optimize(self,
                 base_options_info: Dict[str, Any],
                 repo_paths: List[str],
                 lookups: GeneticAlgorithmLookups,
                 ga_config: OptimizationConfig,
                 file_sample_percentage: float,
                 random_seed: int) -> Dict[str, Any]:
        """
        Abstract method to optimize clang-format configuration.

        Args:
            base_options_info (dict): The initial flat dictionary from clang-format --dump-config.
            repo_paths (list): A list of paths to the temporary git repositories for parallel processing.
            lookups (GeneticAlgorithmLookups): A dataclass containing lookup dictionaries for
                                               option values and forced options.
            ga_config (OptimizationConfig): A dataclass containing configuration parameters
                                            for the optimization algorithm.
            file_sample_percentage (float): Percentage of files to sample for fitness calculation.
            random_seed (int): Seed for random file sampling.

        Returns:
            dict: The flat dictionary of the best clang-format configuration found.
        """
        pass
