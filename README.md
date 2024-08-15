---

<div align="center">    
 
# ITsFlexible: Predicting the conformational flexibility of antibody CDRs

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
[![OpenReview](http://img.shields.io/badge/OpenReview-8C1C12.svg)](https://openreview.net/forum?id=or4tArwd5a)
[![Conference](http://img.shields.io/badge/ICLR-2024-4b44ce.svg)](https://icml.cc/Conferences/2024)
![Unit tests](https://github.com/fspoendlin/AbFlex/actions/workflows/unittest_linting.yml/badge.svg)

</div>


ITsFlexible is a Python package for classifying the conformational flexibility of antibody CDRs and loop motifs in general proteins.
## Abstract

Proteins are highly flexible macromolecules and the ability to adapt their shape is fundamental to many functional properties. While a single, `static' protein structure can be predicted at high accuracy, current methods are severely limited at predicting structural flexibility. A major factor limiting such predictions is the scarcity of suitable training data. Here, we focus on the functionally important antibody CDRs and related loop motifs. We implement a strategy to create a large dataset of evidence for conformational flexibility and develop ITsFlexible, a method able to predict CDR flexibility with high accuracy.

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

```python
from ITsFlexible import classify

classify(infile='path/to/input.csv', outfile='path/to/output.csv')
```

The input file should be a csv file with the following columns:

| index | pdb | ab_chains | chain | resi_start | resi_end |
| ----- | --- | --------- | ----- | ---------- | -------- |
| 0     | /path/to/structure.pdb | labels of all chains included in context (i.e. heavy & light chain) | chain with loop | first residue of loop | last residue of loop |


## Citation


The codebase is inspired by ![Graphinity](https://github.com/amhummer/Graphinity) and ![egnn](https://github.com/vgsatorras/egnn).
