from typing import List, Dict, Any
from fastfeature.transformations.Transformation import Transformation
import numpy as np

class CandidateFeature:
    def __init__(self, transformation: Transformation, parents: List['CandidateFeature']):
        self.transformation = transformation
        self.name = ''
        self.parents: List[CandidateFeature] = parents # candidate features that are needed for the transformation
        self.transformed_matrices = {}

        self.depth = None
        self.number_of_transformations = None
        self.number_of_raw_attributes = None


        self.derive_properties()


    def derive_properties(self):
        self.properties = {}
        # type properties
        self.properties['type'] = str('float64')


    def get_name(self):
        if self.name == '':
            name_list: List[str] = []
            for p in self.parents:
                name_list.append(p.get_name())
            self.name = self.transformation.get_name(name_list)
        return self.name

    def materialize(self):
        if len(self.transformed_matrices) != 0:
            return self.transformed_matrices

        #materialize parents
        materialized_matrices: Dict[str, Any] = {}
        for p_i in range(len(self.parents)):
            materialization: Dict[str,Any] = self.parents[p_i].materialize()
            for k, v in materialization.items():
                if p_i == 0:
                    materialized_matrices[k] = np.array(materialization[k]).reshape(-1, 1)
                else:
                    #print(self.parents[p_i].get_name())
                    #print(str(np.array(materialized_matrices[k]).shape) + " vs " + str(np.array(materialization[k]).shape) )
                    materialized_matrices[k] = np.hstack((materialized_matrices[k], np.array(materialization[k]).reshape(-1, 1)))
                    #print(materialized_matrices[k].shape)


        #fit transformation to training
        self.transformation.fit(materialized_matrices['train'])

        self.transformed_matrices = {}
        for k in materialized_matrices.keys():
            self.transformed_matrices[k] = self.transformation.transform(materialized_matrices[k])

        return self.transformed_matrices

    def get_transformation_depth(self):
        if self.depth == None:
            depths: List[int] = []
            for p in self.parents:
                depths.append(p.get_transformation_depth() + 1)
            self.depth = max(depths)
        return self.depth

    def get_number_of_transformations(self):
        if self.number_of_transformations == None:
            self.number_of_transformations: int = 0
            for p in self.parents:
                self.number_of_transformations += p.get_number_of_transformations() + 1
        return self.number_of_transformations


    #not the unique set, but the number
    def get_number_of_raw_attributes(self):
        if self.number_of_raw_attributes == None:
            self.number_of_raw_attributes: int = 0
            for p in self.parents:
                self.number_of_raw_attributes += p.get_number_of_raw_attributes()
        return self.number_of_transformations

    def __repr__(self):
        return self.get_name()


    # less complexity than other feature
    def __lt__(self, other: 'CandidateFeature'):
        #first, compare depth -> tree depth
        if self.get_transformation_depth() < other.get_transformation_depth():
            return True
        if self.get_transformation_depth() > other.get_transformation_depth():
            return False

        # second, the number of transformations -> number of nodes that are not leaves
        if self.get_number_of_transformations() < other.get_number_of_transformations():
            return True
        if self.get_number_of_transformations() > other.get_number_of_transformations():
            return False

        # third, the number of raw attributes -> number of leaves
        if self.get_number_of_raw_attributes() < other.get_number_of_raw_attributes():
            return True
        if self.get_number_of_raw_attributes() > other.get_number_of_raw_attributes():
            return False



        '''
        - Think about how nesting affects complexity, e.g., normalize(discretize(A)) vs discretize(normalize(A)
        - Create transformation type complexity hierarchy, e.g. A + B < Group A By B Then Max
        - Think about feature value range, e.g. |N < |R
        
        Other ideas:
        - think about more statistics for trees, e.g., Height is the length of the longest path to a leaf
        
        '''

        return False
