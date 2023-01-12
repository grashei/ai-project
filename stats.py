from typing import List
from graph import Graph


def print_data_stats(graphs: List[Graph]):
    family_sets = {}
    for graph in graphs:
        for part in graph.get_parts():
            if part.get_family_id() not in family_sets.keys():
                family_sets[part.get_family_id()] = set()
            family_sets.get(part.get_family_id()).add(part.get_part_id())

    no_parts = 0
    for fam, parts in family_sets.items():
        no_parts += len(parts)
        for fam_intersect, parts_intersect in family_sets.items():
            if fam != fam_intersect:
                intersection = parts.intersection(parts_intersect)
                if intersection:
                    print(f'{fam} intersects with {fam_intersect} on {intersection}')

    family_ids = {part.get_family_id() for graph in graphs for part in graph.get_parts()}
    part_ids = {part.get_part_id() for graph in graphs for part in graph.get_parts()}