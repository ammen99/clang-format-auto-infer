# Clang-Format Optimizer 🚀
This project provides a tool for quickly configuring `clang-format` to match the style of an existing codebase. In other words, it aims to find a `.clang-format` configuration that minimizes the number of changes (insertions + deletions) when applied to the codebase, thereby reducing
formatting churn and improving code consistency.

## Features
*   **Genetic Algorithm Optimization** 🧬: Employs a genetic algorithm with an island model to explore a wide range of `clang-format` options and find an optimal configuration.
*   **Black-Box Optimization with Nevergrad** 📦: Integrates Nevergrad, a state-of-the-art black-box optimization library, offering alternative algorithms like CMA-ES, Differential Evolution, and more.
*   **Abstracted Optimization Strategies**: The core optimization logic is now abstracted, allowing for easy integration of alternative optimization algorithms in the future.
*   **Automatic Detection of `clang-format` Options** 🔍: Includes a tool to parse the latest `clang-format` documentation from the web and extracts information about available options. The result is stored in a human-readable JSON file which can be further tweaked to exclude certain options from the optimization process. Options extracted from clang-format 20 can be found in `data/clang-format-values.json`.
*   **Forced Options** 🔒: Allows users to specify certain `clang-format` options that should always be set to a particular value, overriding the optimization process for those specific options. See `data/forced.yml` as an example of the options you would typically configure by hand.
*   **Change Minimization** 📊: The fitness function for the genetic algorithm is based on minimizing the `git diff --shortstat` output (total insertions and
deletions) after applying `clang-format`.
*   **Interactive Plotting** 📈: Optionally visualizes the best fitness score over time for each island using `matplotlib` (use `--plot-fitness` to enable).
*   **Parallelization** ⚡: Utilize multiple CPU cores by creating temporary copies of the repository and running fitness calculations in parallel.
*   **File Sampling** ✂️: Speed up optimization by randomly sampling a percentage of files from the repository for fitness evaluation.
*   **Graceful Termination** 🛑: Supports `Ctrl-C` to gracefully stop the optimization process and return the best configuration found so far.
## Installation
### Prerequisites 🛠️
Before you begin, ensure you have the following installed and available in your system's PATH:
*   **Python 3.x** 🐍: The project is developed in Python.
*   **clang-format**: The `clang-format` tool itself, which is part of the LLVM project. You can usually install it via your system's package manager (e.g.,
`sudo apt install clang-format` on Ubuntu, `brew install llvm` on macOS).
*   **Git**: The version control system, used for repository operations and diffing.
### Python Dependencies 📦
It's recommended to use a virtual environment.
1.  **Create a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    ```
2.  **Install the required Python packages:**
    ```bash
    pip install PyYAML requests beautifulsoup4 lxml matplotlib nevergrad
    ```
    *   `PyYAML`: For parsing and generating YAML configurations.
    *   `requests`, `beautifulsoup4`, `lxml`: For scraping `clang-format` documentation to get option values.
    *   `matplotlib`: (Optional) For plotting the fitness history during optimization. If not installed, plotting will be disabled.
    *   `nevergrad`: For black-box optimization algorithms.
## Usage ⚙️
The optimization process involves two main steps: first, generating a JSON file with `clang-format` option values (if the one provided in the repo is not sufficient/outdated), and then running the main optimizer.
### Step 1: (Optional) Generate Clang-Format Option Values 📝
The `get_option_values.py` script scrapes the official `clang-format` style options documentation to create a JSON file containing all known options, their
types, and possible enum values. This file is crucial for the optimizer to know which values to test for each option:

```sh
python3 get_option_values.py > data/clang-format-values.json
```

### Step 2: (Optional, Recommended) Define Forced Options 🔒
If you have certain `clang-format` options that you always want to keep at a specific value (e.g., `ColumnLimit: 120`), you can define them in a YAML file.
Example `data/forced.yml`:

```yml
IndentWidth: 4
UseTab: Never
```

### Step 3: Run the Optimizer 🚀
The `main.py` script is the core of the optimizer. It takes your repository path and the generated option values file as input, then runs the chosen optimization algorithm.

```sh
python3 main.py <repo_path> [OPTIONS]
```

**Arguments:**
*   `<repo_path>`: **Required**. Path to the git repository you want to analyze and optimize the `clang-format` configuration for.
**Options:**
*   `--output <file_path>`: Path to the file where the optimized `.clang-format` configuration will be written. If not provided, the output is printed to
`stdout`.
*   `--option-values-json <file_path>`: Path to the JSON file containing `clang-format` options and their possible values (generated by `get_option_values.py`).
*   `--forced-options-yaml <file_path>`: Path to a YAML file containing options that should be forced to a specific value (e.g., `data/forced.yml`).
*   `-d`, `--debug`: Enable debug output, showing commands being executed and more verbose information.
*   `--optimizer {genetic|nevergrad}`: Choose the optimization algorithm. `genetic` (default) uses a genetic algorithm with an island model. `nevergrad` uses Nevergrad's black-box optimization algorithms.
*   `--iterations <int>`: **[Genetic Algorithm]** Number of generations for the genetic algorithm (default: `100`). More iterations can lead to better results but take longer.
*   `--population-size <int>`: **[Genetic Algorithm]** Total number of individuals across all islands in the genetic algorithm population (default: `4`).
*   `--islands <int>`: **[Genetic Algorithm]** Number of independent populations (islands) for the genetic algorithm (default: `1`). Using more islands can help explore the search space more effectively.
*   `--plot-fitness`: **[Genetic Algorithm]** Visualize the best fitness score over time for each island using `matplotlib`.
*   `--ng-budget <int>`: **[Nevergrad]** Total number of evaluations (budget) for the Nevergrad optimizer (default: `1000`).
*   `--ng-optimizer <str>`: **[Nevergrad]** Name of the Nevergrad optimizer to use (e.g., `OnePlusOne`, `CMA`, `DE`, `PSO`). See Nevergrad documentation for available optimizers.
*   `-j`, `--jobs <int>`: Number of parallel jobs to run for fitness calculation (default: `1`). Each job will operate on a separate temporary copy of your repository. Increase this to utilize more CPU cores.
*   `--start-config-file <file_path>`: Path to an existing `.clang-format` file to use as the initial configuration for optimization. If this option is not provided, the tool will start with the default configuration obtained from `clang-format --dump-config`.
*   `--file-sample-percentage <float>`: Percentage of files (0.0-100.0) to randomly sample from the repository for fitness calculation (default: `100.0`). Lower values speed up the process but may reduce accuracy.

### Example Usage (Genetic Algorithm)
To optimize the `clang-format` configuration for a repository located at `/home/user/my_project`, using the generated JSON values and a forced options YAML,
running for 50 iterations with 4 islands (4 individuals per island), and saving the output to `optimized.clang-format`, using 4 parallel jobs, and sampling 25% of files:

```sh
python3 main.py /home/user/my_project \
    --optimizer genetic \
    --option-values-json data/clang-format-values.json \
    --forced-options-yaml data/forced.yml \
    --iterations 50 \
    --population-size 16 \
    --islands 4 \
    --output optimized.clang-format \
    --plot-fitness \
    --jobs 4 \
    --file-sample-percentage 25.0
```

### Example Usage (Nevergrad)
To optimize the `clang-format` configuration for a repository located at `/home/user/my_project` using the `Nevergrad` optimizer `CMA`, with a budget of 2000 evaluations, and 8 parallel jobs:

```sh
python3 main.py /home/user/my_project \
    --optimizer nevergrad \
    --option-values-json data/clang-format-values.json \
    --forced-options-yaml data/forced.yml \
    --ng-budget 2000 \
    --ng-optimizer CMA \
    --output optimized.clang-format \
    --jobs 8 \
    --file-sample-percentage 50.0
```

To start optimization from an existing `.clang-format` file named `my_base_config.clang-format`:

```sh
python3 main.py /home/user/my_project \
    --start-config-file my_base_config.clang-format \
    --option-values-json data/clang-format-values.json \
    --forced-options-yaml data/forced.yml \
    --iterations 50 \
    --population-size 16 \
    --islands 4 \
    --output optimized.clang-format \
    --plot-fitness \
    --jobs 4
```

## Contributing 👋
Contributions are welcome! Please feel free to open issues or pull requests.

## License 📄
This project is licensed under the terms specified in the `LICENSE` file.
