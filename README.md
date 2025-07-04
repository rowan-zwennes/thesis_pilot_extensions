# Thesis: Extensions to PILOT and RaFFLE Frameworks

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository contains the code and experiments for a bachelor's thesis that replicates and extends the PILOT and RaFFLE machine learning frameworks. The primary goal of this work is to implement and evaluate several novel extensions, including new approaches for feature selection and model ensembling.

All code required to set up the environment, process the data, and reproduce the results presented in the thesis is provided here.

## Relationship to Original Work

This project is built directly upon the foundational research and source code from the following papers. All credit for the core algorithms belongs to the original authors.

**Original Publications:**
> Raymaekers, J., Rousseeuw,P. J., Verdonck, T., & Yao, R. (2024). Fast linear model trees by PILOT. *Machine Learning*, 1-50. https://doi.org/10.1007/s10994-024-06590-3.

> Raymaekers, J., Rousseeuw, P. J., Servotte, T., Verdonck, T., & Yao, R. (2025). A Powerful Random Forest Featuring Linear Extensions (RaFFLE). https://doi.org/10.48550/arXiv.2502.10185

The original source code can be found at the [**STAN-UAntwerp/PILOT GitHub repository**](https://github.com/STAN-UAntwerp/PILOT).

## Codebase Overview

This repository is self-contained. It includes necessary files from the original project, files that have been modified for the new experiments, and completely new files for analysis and data processing.

#### Modified Original Files
The following core files from the PILOT framework were adjusted or extended to implement new functionalities:
- `PILOT.py`: Main algorithm implementation.
- `Tree.py`: Core tree-building logic.
- `download_data.py`: Preprocessing script, adapted for the thesis datasets.
- `benchmark_config.py`: Configuration file for experiments.

#### New Files
These files were created specifically for this thesis to manage the new experimental pipeline, data gathering, and results analysis:
- `benchmark_new.py`: The main script for running all thesis experiments.
- `benchmark_util_new.py`: Utility functions supporting the new benchmark script.
- `Gather_datasets.R`: R script to download and standardize datasets.
- `M5.R`: R script for running the M5 model benchmark.
- `Main_table_code.R`: R script for aggregating results into final tables.
- `NLFS_Plot.R`: R script to generate the time complexity plot.

## Getting Started

### 1. Environment Setup
To ensure full reproducibility, a detailed guide is provided for configuring a Linux virtual machine with all the necessary dependencies, libraries, and specific software versions.

➡️ **Please follow the [Virtual Machine & Environment Setup Guide](SETUP_GUIDE.md) first.**

### 2. Reproducing Experiments
Once the environment is configured, a second guide provides the step-by-step instructions for downloading the data, running the benchmark scripts, and generating the final results and plots.

➡️ **For all experimental steps, see the [Guide to Reproducing Experiments](REPRODUCTION_GUIDE.md).**

## Usage Example

The main experiments are executed using the `benchmark_new.py` script. After setting up the environment and data, you can run experiments from the terminal.

For example, to run the PILOT and Random Forest models on the `boston` and `airfoil` datasets:
```bash
python benchmark_new.py -e MyFirstRun -m PILOT,RF -d boston,airfoil

For a detailed explanation of all arguments and models, please refer to the reproduction guide.

## License

The original PILOT project is licensed under the GPL-3.0 License. In accordance with its terms, this derivative work is also released under the **GNU General Public License v3.0**.

## How to Cite

This repository contains two levels of work: the original frameworks and the novel extensions developed in this thesis. Please cite appropriately.

### Citing the Original Algorithms

If you use the core PILOT or RaFFLE algorithms in your research, it is essential to cite the original publications:

> Raymaekers, J., Rousseeuw, P. J., Verdonck, T., & Yao, R. (2024). Fast linear model trees by PILOT. *Machine Learning*, 1-50. https://doi.org/10.1007/s10994-024-06590-3.

> Raymaekers, J., Rousseeuw, P. J., Servotte, T., Verdonck, T., & Yao, R. (2025). A Powerful Random Forest Featuring Linear Extensions (RaFFLE). https://doi.org/10.48550/arXiv.2502.10185

### Citing This Thesis Work

If you are specifically referencing the extensions, results, or code from this thesis, please cite this repository. The preferred method is to use the "Cite this repository" button on the right-hand side of the main repository page.

Alternatively, you can use the following BibTeX entry for your thesis:

```bibtex
@mastersthesis{Zwennes2025pilot_extensions,
  author    = {Rowan Zwennes},
  title     = {Enhancing PILOT Trees with Sparse Multivariate Models via Integrated Regularization},
  school    = {Erasmus University},
  year      = {2025},
  note      = {Code available at: \url{https://github.com/[your-github-username]/thesis-pilot-extensions}}
}
