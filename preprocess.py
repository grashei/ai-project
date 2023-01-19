from torch import Tensor

from graph import Graph
import torch
from typing import Dict, List, Set, Tuple


def create_family_id_mapping(graphs: List[Graph]) -> Dict:
    family_ids = {int(part.get_family_id()) for graph in graphs for part in graph.get_parts()}
    family_ids_list = sorted(list(family_ids))
    mapping = {}
    for idx, fam_id in enumerate(family_ids_list):
        mapping[fam_id] = idx

    return mapping


def create_features_from_graph(graph: Graph, fam_mapping: Dict) -> Tuple[List[Tensor], List[int]]:
    feature_size = len(fam_mapping)

    # Create a list of existing nodes in graph
    existing_nodes = [int(part.get_family_id()) for part in graph.get_parts()]

    # Create a tensor for existing nodes and count their occurrences
    existing_nodes_tensor = torch.zeros(len(fam_mapping), dtype=torch.uint8)
    for node in existing_nodes:
        existing_nodes_tensor[fam_mapping[node]] += 1

    existing_edge_list = []

    # Create tensors for every edge in the graph of the form (from_part_fam_id, to_part_fam_id)
    for node, edges in graph.get_edges().items():
        from_node = fam_mapping[int(node.get_part().get_family_id())]
        for edge in edges:
            to_node = fam_mapping[int(edge.get_part().get_family_id())]
            if from_node < to_node:
                existing_edge_list.append([from_node, to_node])

    # Create all possibly existing edges
    existing_nodes_set = set(existing_nodes)
    not_existing_edge_list = []
    for from_node in existing_nodes_set:
        for to_node in existing_nodes_set:
            if from_node < to_node:
                edge = [fam_mapping[from_node], fam_mapping[to_node]]
                if edge not in existing_edge_list:
                    not_existing_edge_list.append(edge)

    edge_list = existing_edge_list + not_existing_edge_list

    edge_tensor_list = []
    for edge in edge_list:
        from_node, to_node = edge
        edge_tensor = torch.zeros(feature_size, dtype=torch.uint8)
        edge_tensor[from_node] += 1
        edge_tensor[to_node] += 1
        edge_tensor = torch.cat((edge_tensor, existing_nodes_tensor), dim=-1)
        edge_tensor_list.append(edge_tensor)

    targets = [1] * len(existing_edge_list) + [0] * len(not_existing_edge_list)

    return edge_tensor_list, targets


def create_features_for_graph_list(graphs: List[Graph]) -> List[Tuple[Tensor, Tensor]]:
    pass
