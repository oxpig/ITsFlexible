---

<div align="center">    
 
# ITsFlexible: Predicting the conformational flexibility of antibody and TCR CDRs

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
[![OpenReview](http://img.shields.io/badge/OpenReview-8C1C12.svg)](https://openreview.net/forum?id=or4tArwd5a)
[![Conference](http://img.shields.io/badge/ICLR-2024-4b44ce.svg)](https://icml.cc/Conferences/2024)
![Unit tests](https://github.com/fspoendlin/ITsFlexible/actions/workflows/unittest_linting.yml/badge.svg)

</div>


ITsFlexible is a Python package for classifying the conformational flexibility of antibody and TCR CDR3s and loop motifs with identical secondary structures (loops bounded between two antiparallel beta-strands) across all proteins. Loops are labelled as flexible if they are predicted to adopt multiple conformations and rigid if they are predicted to adopt a single conformation. A conformations is here defined as an ensemble of structures with a maximum RMSD of 1.25 Å between any two structures.

## Abstract

Many proteins are highly flexible macromolecules and the ability to adapt their shape can be fundamental to their functional properties. Recent advances now mean we can predict a single, static protein structure with high accuracy. However, we are not yet able to reliably predict structural flexibility. A major factor limiting such predictions is the scarcity of suitable training data. Here, we focus on the structural flexibility of the functionally important antibody and T-cell receptor CDR3s. We additionally consider CDR3-related loops across all proteins to increase the amount of available data. Structurally, CDR3s are loops bounded by two antiparallel $\beta$-strands. By extracting all such loop motifs from the PDB we create ALL-conformations. The dataset contains 1.2 million examples with more than 100,000 unique sequences and captures all experimentally observed conformations of these loop motifs including antibody and TCR CDR3s. Building on this dataset, we develop ITsFlexible a method that classifies CDR3 flexibility, in ensembles of crystal structures and MD simulations, with high accuracy and outperforms all alternative approaches. We selected 3 antibodies with no previously solved structures and determined accessible CDRH3 conformations using cryo-EM. These experiments confirm the predicted flexibility of two antibodies with high confidence predictions. ALL-conformations and ITsFlexible contribute towards a better understanding of the flexibility in functionally important CDRs and show how ML methods can help to predict protein conformational ensembles.

## Installation

Create a conda environment:

```bash
conda create -n ITsFlexible_env python=3.10
conda activate ITsFlexible_env
```

Install pytorch 2.3 with the appropriate version for your system, (see [pytorch.org](https://pytorch.org/get-started/locally/)). For cpu only, use:

```bash
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
```

Clone the repository and install the package:

```bash
git clone git@github.com:fspoendlin/ITsFlexible.git
cd ITsFlexible
pip install .
```

Install torch geometric for the correct version of your system (see [pytorch-geometric.org](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)). For cpu only, use:

```bash
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
```

Install conda dependencies:

```bash
conda install -c conda-forge openbabel
```

## Usage

**Inference**

To classify CDRs with ITsFlexible you will need to provide a csv file and have crystal or predicted structures of your antibodies/TCRs in pdb format.

The input csv file with the following columns:

| index | pdb | ab_chains | chain | resi_start | resi_end |
| ----- | --- | --------- | ----- | ---------- | -------- |
| 0     | /path/to/structure.pdb | labels of all chains included in context (i.e. heavy & light chain) | chain with loop | first residue included in loop | last residue included in loop |

The CDRs can be classified using the provided script:

```bash
cd scripts
python predict.py --dataset path/to/dataset.csv --predictor loop
```

or by using the following python code:

```python
from ITsFlexible import classify

classify(infile='path/to/input.csv', outfile='path/to/output.csv', predictor='loop')
```

ITsFlexible provides two predictors for CDR flexiblity: `loop` and `anchors`. These differ in the way in which structural similarity is defined. For `loop` similarity is calculated by alignment on the loop residues themselves, while `anchors` similarity is calculated by alignment on the Fv residues (flanking the loop). For the `loop` predictor we recommend setting `resi_start` to IMGT residue 107 and `resi_end` to 116. For the `anchors` predictor we recommend setting `resi_start` to 105 and `resi_end` to 118. If input structures are not IMGT numbered the suggested numbers should be changed to point to the residues corresponding to the specified IMGT residues.

ITsFlexible outputs a csv file with an additional column containing the predicted classification score. Classification scores should be interpreted as follows for the `loop` predictor:

| high confidence rigid | low confidence rigid | ambiguous |low confidence flexible | high confidence flexible |
| ---------------------- | --------------------- | ------------------------ | ------------------------ | ------------------------ |
| 0 - 0.025              | 0.025 - 0.075         | 0.075 - 0.25             | 0.25 - 0.30              | 0.25 - 1                 |

and for the `anchors` predictor:

| high confidence rigid | low confidence rigid | ambiguous |low confidence flexible | high confidence flexible |
| ---------------------- | --------------------- | ------------------------ | ------------------------ | ------------------------ |
| 0 - 0.20               | 0.20 - 0.40           | 0.40 - 0.60              | 0.60 - 0.85              | 0.85 - 1                 |


**Training**

A training script to retrain the ITsFlexible models is provided. The training script requires to download the PDB and CDR datasets, this can be done using scripts in `scripts/downloads/`. File paths in the `data/*csv` files may have to be adjusted depending on the where the files are saved. Wandb logging can be set up by specifiying parameters in the `ITsFlexible/trained_models/config*.yaml` files.

```bash
cd scripts
bash downloads/pdb_download.sh
bash downloads/CDR_test_download.sh
python train.py --predictor loop
```

## Citation

```
@article{Spoendlin2024,
	title = {AbFlex: Predicting the conformational flexibility of antibody CDRs},
	author = {Fabian C. Spoendlin, Wing Ki Wong, Guy Georges, Alexander Bujotzek, and Charlotte M. Deane},
	conference = {1st Machine Learning for Life and Material Sciences Workshop at ICML 2024},
	url = {https://openreview.net/forum?id=or4tArwd5a},
	year = {2024},
}
```

The codebase is inspired by ![Graphinity](https://github.com/amhummer/Graphinity) and ![egnn](https://github.com/vgsatorras/egnn).
