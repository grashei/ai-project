# Project Structure
Most of our code and documentation, including explanations of our decisions, can be found in the `neighbour_prediction_ffn.ipynb` Jupiter Notebook. To evaluate our predictor, please run the Jupitery Notebook file. It only trains our models if the model hasn’t been persisted yet. Since we added the persisted model to our repository, it will automatically skip training.

Note: We slightly modified `evaluation.py` to include the normalized edge accuracy (see the functions `normalized_evaluate_graphs` and `normalized_relative_edge_accuracy`) and by stopping calculating the edge accuracy if the method has to check against at least 10k permutations (the permutation limit is a function parameter, and it can only worsen the edge accuracy).

# Our Approach
![PreidctorDescription](https://user-images.githubusercontent.com/9284845/218280501-d55bd0ad-004d-4a64-8b04-c9350ea43842.jpg)

## The Predictor
Our predictor consists of two parts:
- The model _KantenKennerKarl_ (Karl) predicts the neighbor edges for each node
- The graph-building algorithm _GraphGuruGünter_ (Günter) builds the spanning tree based on Karl's predictions

We built two variants of Karl and Günter, one that works with the nodes' FamilyIDs and one with their PartIDs. Surprisingly they performed almost the same. In the illustration and readme file, ID refers to the node's Family ID or part ID, depending on the variations of Karl and Günter.

## Pros and Cons of Karl & Günter
### Their Advantages 
- They can handle node sets/graphs of all sizes
- Short training duration (the family ID variant takes less than one minutes on a mid-tier graphic card), which allows for rapid hyperparameter tweaking and testing algorithmic changes 
- Always outputs a connected non-cyclomatic graph
- They are tuned with our "normalized accuracy" to also perform well on larger graphs
### Their Weaknesses
- Karl doesn't learn based on Günters accuracy
- Karl only predicts if a node has a neighbor with a specific ID, it doesn't tell Günter how many neighbors with that ID said node has

# Our Accuracy

## Normalized Edge Accuracy
Large graphs are the most tricky ones to predict. To our advantage, the edge accuracy only lightly punishes mispredicted large graphs. We receive a high edge accuracy by default for large spanning trees, regardless of whether the graph represents the original graph (e.g., 81% edge accuracy for a graph containing 20 nodes even though all of the prediction's #node-1 edges are not present in the original graph). A random spanning tree predictor gets 66% accuracy on the dataset on average. We counterbalance with normalizing the edge accuracy: a normalized edge accuracy of 0% indicates the edge accuracy couldn't be worse (assuming that both the predicted and actual graphs are spanning trees), and 100% means the edge accuracy couldn't be better. We optimized our predictor against the normalized edge accuracy which strongly correlates with the default edge accuracy.

## Results
Our best model has an accuracy (against our test set) of 98.8% and normalized accuracy of 97.6%.

# Experiments
These tweaks all didn’t make it in our final model and algorithm as they performed worse with them:
## Single-Edge Prediction
Our original idea was that instead of predicting all neighbors for one node, only predict if there's an edge between two nodes. We dropped the idea because the number of needed edge predictions scales quadratically with graph size while the number of required neighbor predictions scales linearly. We'd need much more edge predictions for large graphs than neighbor predictions, which likely hurts the overall quality of the predicted spanning tree.
## Predict the Number of Neighbours With the Same IDs
Karl only predicts the IDs of a node's neighbors, not if it has multiple neighbors with the same ID and how many. Since Karl's task changed from a classification problem to a classification and regression problem, we removed the sigmoid at the last layer and replaced the BCE Loss with the MSE Loss. The changes resulted in a decrease in both Karl's and Günter's accuracies.
## Tree-Based Approach
Most graphs seem to have a root from which other parts extend symmetrically; it resembles the construction's base. For example, only one node usually has multiple neighbors with the same part ID, and often, this node also has the highest degree. We wanted to see if treating the root node differently and building the output graph around the root node positively affects our accuracy. We looked at the output graph as a tree and defined the root as the node with the highest degree in the center set (choose one at random if multiple have the same degree) and the center set as the set of all vertices of minimum eccentricity. We now built a second fully-connected feed-forward network to predict the graph's root node. It had a similar validation accuracy as Karl (> 99.9%). Then we changed Karl to only predict the children's neighbors instead of all neighbors. Also, we added one dimension to the feature vector that tells Karl if the node for which Karl should predict its neighbors is a root node. Günter now takes these predictions and begins building the graph, starting at the (predicted) root node. We allowed Günter to only connect several parts with the same part id to the root node. We achieved a validation edge accuracy of slightly below 98% with that approach which is lower than the one achieved with our simpler variants of Karl and Günter a few days after. This approach might have been too overengineered; we possibly tried too hard searching for patterns.

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
