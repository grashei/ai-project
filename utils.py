import pickle
from typing import List

from graph import Graph


def load_graphs() -> List[Graph]:
    with open('data/graphs.dat', 'rb') as file:
        return pickle.load(file)
