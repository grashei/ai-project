from graph import Graph
from typing import Dict, List


class FamilyIdMapper:
    def __init__(self, graphs: List[Graph]):
        self.__family_ids = sorted(list({int(part.get_family_id()) for graph in graphs for part in graph.get_parts()}))
        self.__mapping_len = len(self.__family_ids)
        self.dense_mapping = self.__create_dense_family_id_mapping()
        self.orig_mapping = self.__create_orig_family_id_mapping()

    def __create_dense_family_id_mapping(self) -> Dict:
        mapping = {}
        for idx, fam_id in enumerate(self.__family_ids):
            mapping[fam_id] = idx

        return mapping

    def __create_orig_family_id_mapping(self) -> Dict:
        mapping = {}
        for idx, fam_id in enumerate(self.__family_ids):
            mapping[idx] = fam_id

        return mapping

    def to_dense(self, fam_id: int) -> int:
        return self.dense_mapping[fam_id]

    def to_orig(self, fam_id: int) -> int:
        return self.orig_mapping[fam_id]

    def get_len(self) -> int:
        return self.__mapping_len

