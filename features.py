import torch
from typing import List
from typing import Set
from part import Part
from mapping import FamilyIdMapper, PartIdMapper
from torch import Tensor


def create_features_family_id(parts: Set[Part], fam_mapping: FamilyIdMapper) -> List[Tensor]:
    parts_sorted = list(parts)
    parts_sorted.sort()

    num_different_family_ids = len(fam_mapping)

    all_nodes_tensor = torch.zeros(num_different_family_ids, dtype=torch.float)

    for part in parts_sorted:
        dense_family_id = fam_mapping.to_dense(int(part.get_family_id()))
        all_nodes_tensor[dense_family_id] += 1.0

    feature_tensors = []

    for part in parts_sorted:
        dense_family_id = fam_mapping.to_dense(int(part.get_family_id()))
        given_node_tensor = torch.zeros(num_different_family_ids, dtype=torch.float)
        given_node_tensor[dense_family_id] = 1
        feature_tensor = torch.cat((all_nodes_tensor, given_node_tensor), dim=-1)
        feature_tensors.append(feature_tensor)

    return feature_tensors


def create_features_part_id(parts: Set[Part], part_mapper: PartIdMapper) -> List[Tensor]:
    parts_sorted = list(parts)
    parts_sorted.sort()

    num_different_part_ids = len(part_mapper)

    all_nodes_tensor = torch.zeros(num_different_part_ids, dtype=torch.float)

    for part in parts_sorted:
        dense_part_id = part_mapper.to_dense(int(part.get_part_id()))
        all_nodes_tensor[dense_part_id] += 1.0

    feature_tensors = []

    for part in parts_sorted:
        dense_part_id = part_mapper.to_dense(int(part.get_part_id()))
        given_node_tensor = torch.zeros(num_different_part_ids, dtype=torch.float)
        given_node_tensor[dense_part_id] = 1
        feature_tensor = torch.cat((all_nodes_tensor, given_node_tensor), dim=-1)
        feature_tensors.append(feature_tensor)

    return feature_tensors
