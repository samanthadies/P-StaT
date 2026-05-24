# Representational and Behavioral Stability of Truth in Large Language Models

This repository introduces the `P-StaT` (Pertrubation Stability of Truth) framework used to measure
representational and behavioral satability in large language models (LLMs) with
controlled truth-label perturbations.\
It includes scripts for:

-   Extracting **layerwise activations** from Hugging Face models
-   Generating **noise activations**
-   Training **linear probes** (sAwMIL and Mass Mean) for "True
    vs. Not-True" classification
-   Running **zero shot** experiments for "True vs. Not-true" classification
-   Generating all plots in the paper

------------------------------------------------------------------------

### Environment Setup

You can recreate the exact environment using:

``` bash
micromamba env create -f environment.yml
# or
conda env create -f environment.yml
```

To activate the environment on your own system:

``` bash
conda activate stability
```

### HuggingFace Access Tokens

To get HuggingFace **Access Tokens** for gated models, visit 
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
You will need to update the ```configs/model``` files with the token
or pass it into the ```token``` field through the command line to generate
activations with the gated models.

------------------------------------------------------------------------

## Usage and Examples

We use ```Hydra``` to run and manage our experiments. Refer to the
[Hydra documentation](https://hydra.cc/docs/intro/) for help.

### 1. Collect Hidden Activations

The LLM activations are ***not*** included in this repository due to their size.
You can generate the activations for each LLM (e.g., ```llama-3-8b```)
by running

``` bash
python collect_activations.py \
  --config-path configs \
  --config-name activations \
  hydra.run.dir=. \
  model='LLM' \
  model.token='YOUR_HF_TOKEN_HERE' \
  output_dir='outputs/activations/${model.name}'
```

This stores the generated activations for all datasets and saves them at,
for example, [outputs/activations/](outputs/activations/). To generate
activations for a subset of the datasets, include the ```datasets=[]```
hydra override.

### 2. Generate Noise Activations

For each datapack (e.g., ```cities_loc```) and LLM (e.g., ```llama-3-8b```), 
generate the noise activations by running

``` bash
python collect_noise_activations.py \
  --config-path configs \
  --config-name generate_noise \
  hydra.run.dir=. \
  model='LLM' \
  datapack='DATAPACK'
```
This stores noise activations in the same directory as the real activations
and generates dummy noise datasets. To change the number of noise statements 
generated, use the hydra override ```pct_of_train_tag```. This generates noise 
as a proportion of the number of non-noise statements.

### 3. Collect Token Probabilities (Familiarity Analysis)

To compute token-level next-token probabilities, log-probabilities, and
surprisal values for a model and dataset combination (e.g., ```llama-3-8b```), run
``` bash
python compute_token_probabilities.py \
  --config-path configs \
  --config-name token_probabilities \
  hydra.run.dir=. \
  model='LLM'
```

This script computes reusable token-level statistics for all statement
types, and outputs are saved to ```outputs/perplexity/{LLM}/```.

### 4. Compute Within/Between Activation Distances

To compute within-condition and between-condition activation distances
for the familiarity analysis (Figure 2), run

``` bash
python analyze_within_between_activations.py \
  --config-path configs \
  --config-name within_between \
  hydra.run.dir=. \
  model='LLM'
```

The outputs are saved to ```outputs/analysis_data/within_between/```.

### 5. Run Stability Experiments

The stability experiments in the paper involve (1) training baseline ***True vs. 
Not True*** probes and five perturbations and (2) running baseline and perturbed
***True vs. Not True*** zero-shot experiments.

The perturbation type is controlled with the ```task``` parameter as follows:
* ***True vs. Not True***: ```task=0```
* ***True + Synthetic (TF) vs. Not True***: ```task=1```
* ***True + Synthetic (Fi) vs. Not True***: ```task=2```
* ***True + Fictional vs. Not True***: ```task=3```
* ***True + Fictional (T) vs. Not True***: ```task=4```
* ***True + Noise vs. Not True***: ```task=5```

(1) To train the probes, run the following command for each probe, LLM,
and dataset combination, and perturbation type (e.g., ```sAwMIL``` + 
```llama-3-8b``` + ```cities_loc``` + ```0```):

``` bash
python exp_probe_linear.py \
  --config-path=configs \
  --config-name=probe_linear_mil \
  task='TASK' \
  model='LLM' \
  datapack='DATAPACK' \
  probe.name='PROBE' \
  output_dir='outputs/probes/${probe.name}/${model.name}'
```

**Note:** To switch between the ```sAwMIL``` and ```Mass-Mean``` probes,
you must switch both ```probe.name``` and ```config-name``` (```probe_linear_mil```
for ```sAwMIL``` and ```probe_linear_sil``` for ```mean_diff```).

(2) To run the zero-shot experiments, run the following command for each LLM, 
Dataset, and perturbation combination (e.g., ```cities_loc``` + ```llama-3-8b``` + 
```0```):

```bash
python exp_zero_shot.py \
  --config-path=configs \
  --config-name=zero_shot \
  datasets='[DATAPACK]' \
  perturbation_type='TASK' \
  model='MODEL'
```

All artifacts of the trained probes are saved in [outputs/probes/](outputs/probes/).

### 6. Evaluate Stability & Generate Plots

Once all activations are generated and experiments have been run, you can evaluate the 
stability and generate paper plots by running

``` bash
python analyze_stability_and_plot.py
```

This script sequentially (1) builds combined probing prediction summary CSVs for
sAwMIL and Mass-Mean probes, (2) builds combined zero-shot prediction summary CSVs,
(3) builds retraction/expansion summary tables, and (4) recreates all paper figures 
and supplementary figures, (Figure 2, Figure 3 , supplementary robustness plots, and
LLM-level retraction-rate plots).

Summary dataframes are saved in [outputs/analysis_data/](outputs/analysis_data/)) and plots are saved in 
[outputs/plots/](outputs/plots/)).

### 7. Cluster Retractions

To cluster statement-level activations and analyze whether epistemic
retractions are geometrically structured in activation space, run

``` bash
python cluster_retraction_activations.py \
  --dataset DATASET \
  --models default
```

Outputs are written to ```outputs/retraction_activation_clusters/```.

------------------------------------------------------------------------

### **Citations**

1. Savcisens, G. & Eliassi-Rad, T. Trilemma of Truth in Large Language Models, ***Mechanistic Interpretability Workshop at NeurIPS 2025***, [https://openreview.net/forum?id=z7dLG2ycRf](https://openreview.net/forum?id=z7dLG2ycRf) (2025).
2. Marks, S. & Tegmark, M. The Geometry of Truth: Emergent Linear Structure in Language Model Representations of True/False Datasets. ***Proceedings of the 1st Conference on Language Modeling (COLM)***, [https://openreview.net/forum?id=aajyHYjjsk](https://openreview.net/forum?id=aajyHYjjsk) (2024).
