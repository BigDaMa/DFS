from typing import List, Dict, Any
from fastsklearnfeature.transformations.Transformation import Transformation
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from fastsklearnfeature.configuration.Config import Config
import copy
import time
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation


class CandidateFeature:
    def __init__(self, transformation: Transformation, parents: List['CandidateFeature']):
        self.transformation = transformation
        self.name = ''
        self.parents: List[CandidateFeature] = parents # candidate features that are needed for the transformation

        self.depth = None
        self.number_of_transformations = None
        self.number_of_raw_attributes = None

        self.runtime_properties: Dict[str, Any] = {}
        self.score = None #deprecated

        self.pipeline = self.create_pipeline()

        self.derive_properties()


    def create_pipeline(self):
        #parent_features = FeatureUnion([(p.get_name(), p.pipeline) for p in self.parents], n_jobs=Config.get('feature.union.parallelism'))

        if bool(Config.get('pipeline.caching')):
            parent_features = FeatureUnion([(p.get_name() + str(time.time()), p.pipeline) for p in self.parents])
        else:
            parent_features = FeatureUnion([(p.get_name() + str(time.time()), copy.deepcopy(p.pipeline)) for p in self.parents])

        memory = None
        if bool(Config.get('pipeline.caching')):
            memory = "/tmp"

        pipeline = Pipeline([
            ('parents', parent_features),
            (self.transformation.name, self.transformation)
        ], memory=memory)
        return pipeline


    def fit(self, X, y=None):
        return self.pipeline.fit(X)

    def transform(self, X):
        return self.pipeline.transform(X)


    def derive_properties(self):
        self.properties = {}
        # type properties
        self.properties['type'] = str('float64')


    #todo implement this in the transformation class
    def get_name(self):
        if self.name == '':
            name_list: List[str] = []
            for p in self.parents:
                name_list.append(p.get_name())
            self.name = self.transformation.get_name(name_list)
        return self.name

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
                self.number_of_transformations += p.get_number_of_transformations()
        return self.number_of_transformations + 1


    #not the unique set, but the number
    def get_number_of_raw_attributes(self):
        if self.number_of_raw_attributes == None:
            self.number_of_raw_attributes: int = 0
            for p in self.parents:
                self.number_of_raw_attributes += p.get_number_of_raw_attributes()
        return self.number_of_transformations

    def __repr__(self):
        return self.get_name()

    def get_raw_attributes(self):
        raw_attributes: List[RawFeature] = []
        for p in self.parents:
            raw_attributes.extend(p.get_raw_attributes())
        return raw_attributes


    def get_traceability_keys(self, record_i, raw_attributes):
        str_key_list: List[str] = []
        target_value: str = str(self.materialize()['train'][record_i])
        str_key_list.append(target_value)
        for attribute_i in range(len(raw_attributes)):
            str_key_list.append(raw_attributes[0].materialize()['train'][record_i])
        key = tuple(str(element) for element in str_key_list)
        return target_value, key


    '''
    def calculate_traceability(self):
        #first create a tuple for all raw attributes
        raw_attributes = self.get_raw_attributes()

        target_value_count: Dict[Any, int] = {}
        value_combination_count = {}

        for record_i in range(len(raw_attributes[0].materialize()['train'])):
            try:
                target_value, key = self.get_traceability_keys(record_i, raw_attributes)

                if not target_value in target_value_count:
                    target_value_count[target_value] = 0
                target_value_count[target_value] += 1
                if not key in value_combination_count:
                    value_combination_count[key] = 0
                value_combination_count[key] += 1
            except:
                pass

        sum_traceability: float = 0.0
        for record_i in range(len(raw_attributes[0].materialize()['train'])):
            try:
                target_value, key = self.get_traceability_keys(record_i, raw_attributes)
                record_traceability = value_combination_count[key] / float(target_value_count[target_value])
            except:
                record_traceability = 0.0
            sum_traceability += record_traceability

        # return average traceability per record
        avg_traceability = sum_traceability / len(raw_attributes[0].materialize()['train'])

        print(self.get_name() + ": " + str(avg_traceability))

        return avg_traceability
    '''

    def get_complexity(self):
        if self.transformation == None:
            return 1
        elif isinstance(self.transformation, IdentityTransformation):
            return np.sum(np.array([f.get_complexity() for f in self.parents]))
        else:
            return np.sum(np.array([f.get_complexity() for f in self.parents])) + 1



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
        '''
        '''
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
