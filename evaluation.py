import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import permutations
from math import factorial
from typing import Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm

from graph import Graph
from part import Part
from stats import print_data_stats


class MyPredictionModel(ABC):
    """
    This class is a blueprint for your prediction model(s) serving as base class.
    """

    @abstractmethod
    def predict_graph(self, parts: Set[Part]) -> Graph:
        """
        Returns a graph containing all given parts. This method is called within the method `evaluate`.
        :param parts: set of parts to form up a construction (i.e. graph)
        :return: graph
        """


def load_model(file_path: str) -> MyPredictionModel:
    """
        This method loads the prediction model from a file (needed for evaluating your model on the test set).
        :param file_path: path to file
        :return: the loaded prediction model
    """
    with open(file_path, 'rb') as model_file:
        return pickle.load(model_file)


def normalized_evaluate_graphs(model: MyPredictionModel, graphs: List[Graph], show_progress: bool = True) -> float:
    data_set = [(graph.get_parts(), graph) for graph in graphs]
    return normalized_evaluate(model=model, data_set=data_set, show_progress=show_progress)


def evaluate_graphs(model: MyPredictionModel, graphs: List[Graph], max_perms=10_000) -> float:
    data_set = [(graph.get_parts(), graph) for graph in graphs]
    return evaluate(model=model, data_set=data_set, max_perms=max_perms)


def normalized_evaluate(
        model: MyPredictionModel,
        data_set: List[Tuple[Set[Part], Graph]],
        show_progress: bool = True,
) -> float:
    accuracies = 0.0
    with tqdm(data_set, disable=not show_progress) as data_set_with_progress:
        for input_parts, target_graph in data_set_with_progress:
            predicted_graph = model.predict_graph(input_parts)
            accuracies += normalized_relative_edge_accuracy(predicted_graph, target_graph)
    return accuracies / len(data_set)


def evaluate(
        model: MyPredictionModel,
        data_set: List[Tuple[Set[Part], Graph]],
        max_perms=10_000,
        show_progress: bool = True,
) -> float:
    """
    Evaluates a given prediction model on a given data set.
    :param model: prediction model
    :param data_set: data set
    :return: evaluation score (for now, edge accuracy in percent)
    """
    sum_correct_edges = 0
    edges_counter = 0
    with tqdm(data_set, disable=not show_progress) as data_set_with_progress:
        for input_parts, target_graph in data_set_with_progress:
            predicted_graph = model.predict_graph(input_parts)

            edges_counter += len(input_parts) * len(input_parts)
            sum_correct_edges += edge_accuracy(predicted_graph, target_graph, max_perms)

            # FYI: maybe some more evaluation metrics will be used in final evaluation

    return sum_correct_edges / edges_counter * 100


def normalized_relative_edge_accuracy(predicted_graph: Graph, target_graph: Graph, max_perms=10_000) -> float:
    """
    :return: a value in range [0.0,1.0]. 0.0 means the edge accuracy couldn't be worse (assuming that both graphs are
    spanning trees), 1.0 means the edge accuracy couldn't be better.
    """
    node_count = len(target_graph.get_nodes())
    edge_count = node_count - 1
    # There are #((node_count**2) - edge_count) node combinations that are not connected in the graph. If we mispredict
    # all edges, #edge_count of all possible node combinations don't have an edge, even though they should have one.
    # However, we still get most node combinations that are not connected right; only #edge_count combinations have an
    # edge even though they shouldn't have one. So, 2 * #edge_count node combinations have the wrong edge prediction in
    # total. All other #((node_count ** 2) - (2 * edge_count)) node combinations were predicated correctly. The worst
    # possible percentage is therefore ((node_count ** 2) - (2 * edge_count)) / (node_count ** 2).
    worst_possible_rel_accuracy = 1 - ((2 * edge_count) / (node_count ** 2))
    best_possible_rel_accuracy = 1.0

    actual_rel_accuracy = relative_edge_accuracy(predicted_graph, target_graph, max_perms)
    return inv_lerp(worst_possible_rel_accuracy, best_possible_rel_accuracy, actual_rel_accuracy)


def relative_edge_accuracy(predicted_graph: Graph, target_graph: Graph, max_perms=10_000) -> float:
    """
    :return: the edge accuracy as a value between 0.0 and 1.0 (both inclusive).
    """
    node_count = len(target_graph.get_nodes())
    return edge_accuracy(predicted_graph, target_graph, max_perms) / node_count ** 2


def inv_lerp(a: float, b: float, v: float) -> float:
    """Inverse Linar Interpolation, get the fraction between a and b on which v resides.
    Examples
    --------
        0.5 == inv_lerp(0, 100, 50)
        0.8 == inv_lerp(1, 5, 4.2)
    """
    return (v - a) / (b - a)


def edge_accuracy(predicted_graph: Graph, target_graph: Graph, max_perms=None) -> int:
    """
    Returns the number of correct predicted edges.
    :param predicted_graph:
    :param target_graph:
    :return:
    """
    assert len(predicted_graph.get_nodes()) == len(target_graph.get_nodes()), 'Mismatch in number of nodes.'
    assert predicted_graph.get_parts() == target_graph.get_parts(), 'Mismatch in expected and given parts.'

    best_score = 0

    # Determine all permutations for the predicted graph and choose the best one in evaluation
    perms: List[Tuple[Part]] = __generate_part_list_permutations(predicted_graph.get_parts())

    # Determine one part order for the target graph
    target_parts_order = perms[0]
    target_adj_matrix = target_graph.get_adjacency_matrix(target_parts_order)

    for i, perm in enumerate(perms[:max_perms]):
        # Calculating the accuracy for this graph would take too long, break after max_perms permutations.
        predicted_adj_matrix = predicted_graph.get_adjacency_matrix(perm)
        score = np.sum(predicted_adj_matrix == target_adj_matrix)
        best_score = max(best_score, score)

    return best_score


def calculate_num_permutations(g: Graph) -> int:
    part_occurrence_map = defaultdict(int)
    for part in g.get_parts():
        part_occurrence_map[part.get_part_id()] += 1

    result = 1
    for occurrence in part_occurrence_map.values():
        result *= factorial(occurrence)
    return result


def __generate_part_list_permutations(parts: Set[Part]) -> List[Tuple[Part]]:
    """
    Different instances of the same part type may be interchanged in the graph. This method computes all permutations
    of parts while taking this into account. This reduced the number of permutations.
    :param parts: Set of parts to compute permutations
    :return: List of part permutations
    """
    # split parts into sets of same part type
    equal_parts_sets: Dict[Part, Set[Part]] = {}
    for part in parts:
        for seen_part in equal_parts_sets.keys():
            if part.equivalent(seen_part):
                equal_parts_sets[seen_part].add(part)
                break
        else:
            equal_parts_sets[part] = {part}

    multi_occurrence_parts: List[Set[Part]] = [pset for pset in equal_parts_sets.values() if len(pset) > 1]
    single_occurrence_parts: List[Part] = [next(iter(pset)) for pset in equal_parts_sets.values() if len(pset) == 1]

    full_perms: List[Tuple[Part]] = [()]
    for mo_parts in multi_occurrence_parts:
        perms = list(permutations(mo_parts))
        full_perms = list(perms) if full_perms == [()] else [t1 + t2 for t1 in full_perms for t2 in perms]

    # Add single occurrence parts
    full_perms = [fp + tuple(single_occurrence_parts) for fp in full_perms]
    assert all([len(perm) == len(parts) for perm in full_perms]), 'Mismatching number of elements in permutation(s).'
    return full_perms


# ---------------------------------------------------------------------------------------------------------------------
# Example code for evaluation

if __name__ == '__main__':
    # Load train data
    with open('data/graphs.dat', 'rb') as file:
        train_graphs: List[Graph] = pickle.load(file)

    print_data_stats(train_graphs)
    model_file_path = 'data/karl.dat'
    prediction_model: MyPredictionModel = load_model(model_file_path)

    # For illustration, compute eval score on train data
    instances = [(graph.get_parts(), graph) for graph in train_graphs[:100]]
    eval_score = evaluate(prediction_model, instances)
