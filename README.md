# De-Novo-CASP6-Inhibitors-Design-by-Combining-GRU-Based-RNN-with-ML-Approach
## Introduction:

Due to the potencies in the treatments of neurodegenerative diseases, caspase-6 inhibitors have attracted widespread attentions. Herein, gated recurrent unit (GRU)-based recurrent neural network (RNN) combined with machine learning (ML) was employed to construct generative models of caspase-6 inhibitors. The result showed that GRU-based RNN network can be trained as a good string-based molecular generation model, and the positive rate of the generated inhibitors of caspase-6 was improved by transfer learning and molecular docking-based ligand screening.  

## Requirements:
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
  
Disclaimer: In our work, RNN were first trained to generate a chemical language model on the RDKit canonical SMILES dataset containing 2.4 million molecules from the PubChem database (https://pubchem.ncbi.nlm.nih.gov), where the molecules were restrained to containing between 10 and 100 heavy atoms and the max length was 140. Then, a dataset with 433 active caspase-6 inhibitors was used to fine-tune the pre-trained RNN model. Since GITHUB has a single file limit of 100MB, this demo provides a SMILES dataset with 800 thousand molecules for pre-training RNN generator.
