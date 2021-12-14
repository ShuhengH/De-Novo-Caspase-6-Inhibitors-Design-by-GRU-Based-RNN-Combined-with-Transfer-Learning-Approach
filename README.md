# De-Novo-Caspase-6-Inhibitors-Design-by-GRU-Based-RNN-Combined-with-Transfer-Learning-Approach
## Introduction

Due to the potencies in the treatments of neurodegenerative diseases, caspase-6 inhibitors have attracted widespread attentions. In this demo, gated recurrent unit (GRU)-based recurrent neural network (RNN) combined with transfer learning was used to build the molecular generative model of caspase-6 inhibitors. Furthermore, machine learning (ML)-based predictors were also built for molecular activities prediction.

## Requirements
In order to get started you will need:
  
* Modern NVIDIA GPU, compute capability 3.5 of newer  
* CUDA 9.0  
* Pytorch 0.4.1  
* Scikit-learn  
* RDKit  
* Numpy  
* pandas  
* pickle  
* RDKit  
* tqdm  
* Mordred 
* keras -- for DFFN-based classifier construction

## Installation with Anaconda
If you installed your Python with Anacoda you can run the following commands to get started:

* Create new conda environment with Python 3.6 and activation  
    conda create --new GRU-RNN python=3.6  
    conda activate GRU-RNN  
* Install conda dependencies  
    conda install tqdm  
    conda install -c rdkit -c mordred-descriptor mordred  
    conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch  

## Demos
We uploaded several demos in a form of iPython notebooks:

* Demo_ML-based_classifier.ipynb -- ML-based classifiers construction  
* Demo_Transfer_learning.ipynb -- training a GRU-based RNN to generate new molecules 
* Demo_DFFN-based_classifier -- DFFN-based classifier construction
  
Disclaimer: In our work, RNN were first trained to generate a chemical language model on the RDKit canonical SMILES dataset containing 2.4 million molecules from the PubChem database (https://pubchem.ncbi.nlm.nih.gov), where the molecules were restrained to containing between 10 and 100 heavy atoms and the max length was 140. Then, a dataset with 433 active caspase-6 inhibitors was used to fine-tune the pre-trained RNN model. Since GITHUB has a single file limit of 100MB, this demo provides a SMILES dataset with 800 thousand molecules for pre-training RNN generator.

## Please Cite
Huang, S.H., et al., De Novo Molecular Design of Caspase-6 Inhibitors by a GRU-Based Recurrent Neural Network Combined with a Transfer Learning Approach. Pharmaceuticals 2021, 14 (12), 1249. (https://www.mdpi.com/1424-8247/14/12/1249)
