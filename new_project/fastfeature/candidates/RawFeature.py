from fastfeature.candidates.CandidateFeature import CandidateFeature
from typing import Dict
from typing import Any

class RawFeature(CandidateFeature):
    def __init__(self, name, column_id, splitted_values, properties):
        self.name: str = name
        self.column_id: int = column_id
        self.splitted_values: Dict[str, Any] = splitted_values
        self.properties: Dict[str, Any] = properties
        self.parents = []

    def materialize(self):
        return self.splitted_values

    def get_transformation_depth(self):
        return 0

    def get_number_of_transformations(self):
        return 0

    def get_number_of_raw_attributes(self):
        return 1

    def get_raw_attributes(self):
        return [self]

    def get_name(self):
        return self.name
