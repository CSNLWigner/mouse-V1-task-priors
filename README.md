# Task-specific contextual priors in V1
This repo contains our analysis of [calcium imagining data](https://zenodo.org/records/8109858) from the [Corbo, et al. (2025) Nat.Comm paper](https://doi.org/10.1038/s41467-024-55409-1), which itself built on the orientation-discrimination task and related neural phenomena from the [Corbo, et al. (2022) J.Neurosci](https://doi.org/10.1523/JNEUROSCI.2272-21.2022) paper.

as appeared in
## [TAVAE: A VAE with Adaptable Priors Explains Contextual Modulation in the Visual Cortex](https://arxiv.org/abs/2602.11956v1)

### Abstract

The brain interprets visual information through learned regularities, formalized as performing probabilistic inference under a prior. The visual cortex establishes priors for this inference, some of which are from higher level representations as contextual priors and rely on widely documented top-down connections. While evidence supports that priors are acquired for natural images, it remains unclear if similar separate priors can be flexibly acquired for more specific computations, e.g. when learning a task. To investigate this, we built a generative model trained jointly on natural images and on a simple task, and analyzed it along with large-scale recordings from the early visual cortex of mice. For this, we extended the standard VAE formalism to flexibly and data-efficiently acquire a task such that it reuses representations learned in a task-agnostic manner. The resulting Task-Amortized VAE was used to investigate biases when presenting stimuli that violated the trained task statistics. Such mismatches between the learned task statistics and the incoming sensory evidence resulted in multimodal response profiles, which were also observed in the calcium imaging data from mice performing an analogous task. The task-optimized generative model could account for various characteristics of V1 population activity, including within-day updates to the population responses. Our results confirm that flexible task-specific contextual priors can be learned on-demand by the visual system and can be deployed as early as the entry level of the visual cortex.


### Usage instructions


## Files needed to produce figures in results/iclr_figures_bpteam

### main figure generating scripts
- notebooks/iclr_figures/generatefigurepanels.py
- notebooks/random_visualizations/orientation_activity_map_construction.ipynb

### src (imports used by generatefigurepanels.py)
- src/config.py
- src/data_access.py
- src/data_compilation.py
- src/directional_statistics.py
- src/measure_profile.py
- src/orientation_characterization.py
- src/orientation_representation.py

### matlab files to generate python readable data files from raw data
- matlab/automaticHDF5Conversion.m
- matlab/createHDF5.m
- matlab/decodeByteString.m
- matlab/saveDecodedDataToHDF5.m


## data, see [data-folder-tree.txt](data-folder-tree.txt) file in details:

download data from [https://zenodo.org/records/8109858](https://zenodo.org/records/8109858) and organize into folders provided by [data-folder-tree.txt](data-folder-tree.txt)

### dataset tables and per-experiment files (from BASEDATAPATH in src/config.py)
- data/DATASET1_CellTable.csv
- data/DATASET1_TrialTable.csv
- data/DATASET1_ExperimentTable.csv
- data/DATASET1_CellTrialTable.db
- data/DATASET1_CellTrialTable/\*/\*.h5
- data/DATASET1_CellTrialTable/\*/\*.csv
- data/Naive_CellTable.csv
- data/Naive_TrialTable.csv
- data/Naive_ExperimentTable.csv
- data/Naive_CellTrialTable.db
- data/Naive_CellTrialTable/\*/\*.h5
- data/Naive_CellTrialTable/\*/\*.csv

### data files under repo (read directly in generatefigurepanels.py)
- data/Naive_numpy_arrays/potential_likelihood.npy
- data/Naive_numpy_arrays/task_aggregated_matrices.npy
- data/DATASET1_numpy_arrays/task_aggregated_matrices.npy
- data/Naive_numpy_arrays/tuning_aggregated_matrices.npy
- data/df_frames_widths.csv (provided)
- cache/widths_model.csv (provided)

### generators for data/*_numpy_arrays/*.npy used by generatefigurepanels.py
- notebooks/activity_maps/naive_task_orientation_activity.ipynb
- notebooks/activity_maps/trained_task_orientation_activity.ipynb
- notebooks/activity_maps/naive_tuning_orientation_activity.ipynb
- notebooks/activity_maps/trained_tuning_orientation_activity.ipynb
