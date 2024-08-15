---

<div align="center">    
 
# ITsFlexible: Predicting the conformational flexibility of antibody CDRs

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
[![OpenReview](http://img.shields.io/badge/OpenReview-8C1C12.svg)](https://openreview.net/forum?id=or4tArwd5a)
[![Conference](http://img.shields.io/badge/ICLR-2024-4b44ce.svg)](https://icml.cc/Conferences/2024)
![Unit tests](https://github.com/fspoendlin/ITsFlexible/actions/workflows/unittest_linting.yml/badge.svg)

</div>


ITsFlexible is a Python package for classifying the conformational flexibility of antibody and TCR CDR3s and loop motifs with identical secondary structures in across all proteins. Loops are labelled as flexible if they are predicted to adopt multiple conformations and rigid if they are predicted to adopt a single conformation. A conformations is here defined as an ensemble of structures with a maximum RMSD of 1.25 Å between any two structures.

## Abstract

Proteins are highly flexible macromolecules and the ability to adapt their shape is fundamental to many functional properties. While a single, static protein structure can be predicted at high accuracy, current methods are severely limited at predicting structural flexibility. A major factor limiting such predictions is the scarcity of suitable training data. Here, we focus on the functionally important antibody CDRs and related loop motifs. We implement a strategy to create a large dataset of evidence for conformational flexibility and develop ITsFlexible, a method able to predict CDR flexibility with high accuracy.

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

The CDRs can be classified with the following command:

```python
from ITsFlexible import classify

classify(infile='path/to/input.csv', outfile='path/to/output.csv', predictor='loop')
```

ITsFlexible provides two predictors for CDR flexiblity: `loop` and `anchors`. These differ in the way in which structural similarity is defined. `loop` similarity is calculated by alignment on the loop residues themselves, while `anchors` similarity is calculated by alignment on the Fv residues (flanking the loop). For the `loop` predictor we recommend setting `resi_start` to IMGT residue 107 and `resi_end` to 116. For the `anchors` predictor we recommend setting `resi_start` to 105 and `resi_end` to 118. If input structures are not IMGT numbered the suggested numbers should be changed to point to the residues corresponding to the specified IMGT residues.

ITsFlexible will output a csv file with an additional file containing the predicted probability of a loop being flexible.


**Training**

A training script is provided for ITsFlexible. The training script requires to download the dataset from ..., modify the paths in the `data/*csv` files to point to the correct locations of the pdb files and modify the paths in the `ITsFlexible/trained_models/config*.yaml` files with user defined logging parameters.


## Citation

```
@article{Spoendlin2024,
	title = {AbFlex: Predicting the conformational flexibility of antibody CDRs},
	author = {Fabian C. Spoendlin, Wing Ki Wong, Guy Georges, Alexander Bujotzek, and Charlotte M. Deane},
	conference = {1st Machine Learning for Life and Material Sciences Workshop at ICML 2024},
	url = {https://openreview.net/forum?id=or4tArwd5a},
	year = {2024},
}

The codebase is inspired by ![Graphinity](https://github.com/amhummer/Graphinity) and ![egnn](https://github.com/vgsatorras/egnn).
