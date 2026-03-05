# Representational and Behavioral Stability of Truth in Large Language Models

This repository introduces the `P-StaT` (Pertrubation Stability of Truth) framework used to measure
representational and behavioral satability in large language models (LLMs) with
controlled truth-label perturbations.\
It includes scripts for:

-   Extracting **layerwise activations** from Hugging Face models
-   Generating **noise activations**
-   Training **linear probes** (sAwMIL and Mean Difference) for "True
    vs. Not-True" classification
-   Running **zero shot** experiments for "True vs. Not-true" classification
-   Generating all plots used in the paper (n-gram distributions and activation heatmaps
    decision-boundary heatmaps, stability bar charts)

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

### 3. Run Stability Experiments

The stability experiments in the paper involve (1) training baseline ***True vs. 
Not True*** probes and four perturbations and (2) running baseline and perturbed
***True vs. Not True*** zero-shot experiments.

The perturbation type is controlled with the ```task``` parameter as follows:
* ***True vs. Not True***: ```task=0```
* ***True + Synthetic vs. Not True***: ```task=1```
* ***True + Fictional vs. Not True***: ```task=2```
* ***True + Fictional (T) vs. Not True***: ```task=3```
* ***True + Noise vs. Not True***: ```task=4```

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

**Note:** To switch between the ```sAwMIL``` and ```Mean Difference``` probes,
you must switch both ```probe.name``` and ```config-name``` (```probe_linear_mil```
for ```sAwMIL``` and ```probe_linear_sil``` for ```Mean Difference```).

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

**Note:** The zero-shot experiment does not support the ***True + Noise vs. Not True*** 
perturbation since ***Noise*** does not have a semantic mapping.

All artifacts of the trained probes are saved in [outputs/probes/](outputs/probes/).

### 4. Evaluate Stability & Generate Plots

Once all activations are generated and experiments have been run, you can evaluate the 
stability by running

``` bash
python analyze_stability_and_plot.py \
  --config-path configs \
  --config-name analysis_pipeline \
  hydra.run.dir=. \
  probe='PROBE'
```

This script generates summary dataframes (saved in 
[outputs/analysis_data/](outputs/analysis_data/)) for each dataset and 
regenerates the plots present in our paper (saved in 
[outputs/plots/](outputs/plots/)).

------------------------------------------------------------------------

### **Citations**

1. Savcisens, G. & Eliassi-Rad, T. Trilemma of Truth in Large Language Models, ***Mechanistic Interpretability Workshop at NeurIPS 2025***, [https://openreview.net/forum?id=z7dLG2ycRf](https://openreview.net/forum?id=z7dLG2ycRf) (2025).
2. Marks, S. & Tegmark, M. The Geometry of Truth: Emergent Linear Structure in Language Model Representations of True/False Datasets. ***arXiv preprint arXiv:2310.06824***, [https://arxiv.org/abs/2310.06824](https://arxiv.org/abs/2310.06824) (2024).
