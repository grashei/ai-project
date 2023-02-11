# Our Approach
![PreidctorDescription](https://user-images.githubusercontent.com/9284845/218280501-d55bd0ad-004d-4a64-8b04-c9350ea43842.jpg)

## The Predictor
Our predictor consists of two parts:
- The model _KantenKennerKarl_ (Karl) predicts the neighbor edges for each node
- The graph-building algorithm _GraphGuruGünter_ (Günter) builds the spanning tree based on Karl's predictions

We built two variants of Karl and Günter, one that works with the nodes' FamilyIDs and one with their PartIDs. Supprisingly the performed almost the same. In the illustration, the node's ID refers to its Family ID or part ID, depending on the variations of Karl and Günter.

## Pros and Cons of Karl & Günter
### Their Advantages 
- They can handle node sets/graphs of all sizes
- Short training duration (less than five minutes on budget graphic cards), which allows for rapid hyperparameter tweaking and testing algorithmic changes 
- Always outputs a connected non-cyclomatic graph
- They are tuned with our "normalized accuracy" to also perform well on larger graphs
### Their Weaknesses
- Karl doesn't learn based on Günters accuracy
- Karl only predicts if a node has a neighbor with a specific ID, it doesn't tell Günter how many neighbors with that ID said node has

# Our Accuracy
Large graphs are the most tricky ones to predict. To our advantage, the edge accuracy only lightly punishes mispredicted large graphs. We receive a high edge accuracy by default for large spanning trees, regardless of whether the graph represents the original graph (e.g., 81% edge accuracy for a graph containing 20 nodes even though all of the prediction's #node-1 edges are not present in the original graph). A random spanning tree predictor gets 66% accuracy on the dataset on average. We counterbalance with normalizing the edge accuracy: a normalized edge accuracy of 0% indicates the edge accuracy couldn't be worse (assuming that both the predicted and actual graphs are spanning trees), and 100% means the edge accuracy couldn't be better. We optimized our predictor against the normalized edge accuracy which strongly correlates with the default edge accuracy.

---
# ai-lecture-project

This project is written with Python `3.8` based on Anaconda (https://www.anaconda.com/distribution/).

## Getting started

The file 'requirements.txt' lists the required packages.

1. We recommend to use a virtual environment to ensure consistency, e.g.   
   `conda create -n ai-project python=3.8`

2. Install the dependencies:  
   `conda install -c conda-forge --file requirements.txt`

## Software Tests

This project contains some software tests based on Python Unittest (https://docs.python.org/3/library/unittest.html).
Run `python -m unittest` from the command line in the repository root folder to execute the tests. This should
automatically search all unittest files in modules or packages in the current folder and its subfolders that are
named `test_*`.
