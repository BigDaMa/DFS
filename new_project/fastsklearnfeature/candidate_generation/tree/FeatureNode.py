from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature

class FeatureNode(object):
    def __init__(self, current_feature: CandidateFeature):
        self.current_feature = current_feature
        self.children = []

    def add_child(self, node: 'FeatureNode'):
        self.children.append(node)