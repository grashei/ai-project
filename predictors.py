from graph import Graph
from typing import List
from typing import Set
from typing import Tuple
from features import create_features_family_id, create_features_part_id
from part import Part
import torch
from torch import Tensor
from abc import ABC, abstractmethod


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


class GraphenGuruGuenterFamId(MyPredictionModel):

    def __init__(self, model, fam_mapping):
        self.model = model
        self.fam_mapping = fam_mapping

    def get_parts_with_fam_id(self, parts: List[Part], fam_id: int) -> List[Part]:
        return [part for part in parts if int(part.get_family_id()) == fam_id]

    def predict_graph(self, parts: Set[Part]) -> Graph:
        raw_predictions = []
        feature_tensors = create_features_family_id(parts, self.fam_mapping)
        for feature_tensor in feature_tensors:
            prediction = self.model(feature_tensor)
            raw_predictions.append(prediction)
        raw_predictions = torch.stack(raw_predictions)
        return self.build_predicted_graph(parts, raw_predictions)

    def build_predicted_graph(self, parts: Set[Part], pred_adj_matrix: Tensor) -> Graph:
        parts_list = sorted(list(parts))
        added_parts = set()
        node_count = len(parts_list)
        predicted_graph = Graph()
        parts_at_nodes = dict()
        for p in parts:
            parts_at_nodes[p] = []

        while predicted_graph.get_edge_count() // 2 < node_count - 1:
            assert torch.any(pred_adj_matrix > 0)
            max_signal_idx = (pred_adj_matrix == torch.max(pred_adj_matrix)).nonzero()

            source_idx = max_signal_idx[0][0].item()
            sink_dense_fam_id = max_signal_idx[0][1].item()

            source_orig_fam_id = int(parts_list[source_idx].get_family_id())
            sink_orig_fam_id = self.fam_mapping.to_orig(sink_dense_fam_id)
            source = parts_list[source_idx]

            parts_with_sink_fam_id = self.get_parts_with_fam_id(parts_list, sink_orig_fam_id)
            parts_with_sink_fam_id = sorted(parts_with_sink_fam_id, key=lambda x: parts_at_nodes[x].count(source_orig_fam_id))

            for sink in parts_with_sink_fam_id:
                if (source != sink and not (source in added_parts and sink in added_parts)) or not predicted_graph.is_reachable(source, sink):
                    parts_at_nodes[source] += [sink_orig_fam_id]
                    parts_at_nodes[sink] += [source_orig_fam_id]
                    predicted_graph.add_undirected_edge(source, sink)
                    added_parts.add(source)
                    added_parts.add(sink)
                    break

            pred_adj_matrix[source_idx][sink_dense_fam_id] = 0.0
        return predicted_graph


class GraphenGuruGuenterPartId(MyPredictionModel):

    def __init__(self, model, part_mapping):
        self.model = model
        self.part_mapping = part_mapping

    def get_parts_with_part_id(self, parts: List[Part], part_id: int) -> List[Part]:
        return [part for part in parts if int(part.get_part_id()) == part_id]

    def predict_graph(self, parts: Set[Part]) -> Graph:
        raw_predictions = []
        feature_tensors = create_features_part_id(parts, self.part_mapping)
        for feature_tensor in feature_tensors:
            prediction = self.model(feature_tensor)
            raw_predictions.append(prediction)
        raw_predictions = torch.stack(raw_predictions)
        return self.build_predicted_graph(parts, raw_predictions)

    def build_predicted_graph(self, parts: Set[Part], pred_adj_matrix: Tensor) -> Graph:
        parts_list = sorted(list(parts))
        added_parts = set()
        node_count = len(parts_list)
        predicted_graph = Graph()
        parts_at_nodes = dict()
        for p in parts:
            parts_at_nodes[p] = []

        while predicted_graph.get_edge_count() // 2 < node_count - 1:
            assert torch.any(pred_adj_matrix > 0)
            max_signal_idx = (pred_adj_matrix == torch.max(pred_adj_matrix)).nonzero()
            source_idx = max_signal_idx[0][0].item()
            sink_dense_part_id = max_signal_idx[0][1].item()

            source_orig_part_id = parts_list[source_idx].get_opid()
            sink_orig_part_id = self.part_mapping.to_orig(sink_dense_part_id)
            source = parts_list[source_idx]

            parts_with_sink_part_id = self.get_parts_with_part_id(parts_list, sink_orig_part_id)
            parts_with_sink_part_id = sorted(parts_with_sink_part_id, key=lambda x: parts_at_nodes[x].count(source_orig_part_id))

            for sink in parts_with_sink_part_id:
                if source != sink and not (source in added_parts and sink in added_parts) or not predicted_graph.is_reachable(source,sink):
                    parts_at_nodes[source] += [sink_orig_part_id]
                    parts_at_nodes[sink] += [source_orig_part_id]
                    predicted_graph.add_undirected_edge(source, sink)
                    added_parts.add(source)
                    added_parts.add(sink)
                    break

            pred_adj_matrix[source_idx][sink_dense_part_id] = 0.0
        return predicted_graph