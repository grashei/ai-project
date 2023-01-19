from dataclasses import dataclass

from graph import Graph
from part import Part
from typing import Set


@dataclass(eq=True, order=True, frozen=True)
class UniquePart:
    part_id: int
    family_id: int


def to_unique_part(part: Part) -> UniquePart:
    return UniquePart(part.get_part_id(), part.get_family_id())


def to_unique_parts(parts: Set[Part]) -> Set[UniquePart]:
    return {to_unique_part(p) for p in parts}


def get_unique_parts(graph: Graph) -> Set[UniquePart]:
    return to_unique_parts(graph.get_parts())
