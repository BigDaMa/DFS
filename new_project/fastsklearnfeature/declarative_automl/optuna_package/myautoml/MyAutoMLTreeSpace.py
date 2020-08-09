from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import categorical
import copy
from anytree import Node


class MyAutoMLSpace:
    def __init__(self):
        self.parameter_tree = Node(name='root', parent=None, status=True)
        self.name2node = {}
        self.trial = None

    def generate_cat(self, name, my_list, default_element, depending_node=None):
        parent_node = self.parameter_tree
        if type(depending_node) != type(None):
            parent_node = depending_node

        categorical_node = Node(name=name, parent=parent_node, status=True, default_element=default_element)

        self.name2node[str(name)] = categorical_node
        for element in my_list:
            self.name2node[str(name) + '##' + str(element)] = Node(name=str(name) + '##' + str(element), parent=categorical_node, status=True)

        return categorical_node.children


    def generate_number(self, name, default_element, depending_node=None):
        parent_node = self.parameter_tree
        if type(depending_node) != type(None):
            parent_node = depending_node

        numeric_node = Node(name=name, parent=parent_node, status=True, default_element=default_element)
        self.name2node[str(name)] = numeric_node
        return numeric_node


    def recursive_sampling(self, node, trial):
        for child in node.children:
            if node.status:
                child.status = trial.suggest_categorical(child.name, [True, False])
            else:
                child.status = False
            self.recursive_sampling(child, trial)

    def sample_parameters(self, trial):
        #create dependency graph
        self.recursive_sampling(self.parameter_tree, trial)

    def suggest_int(self, name, low, high, step=1, log=False):
        if self.name2node[str(name)].status:
            return self.trial.suggest_int(name, low, high, step=step, log=log)
        else:
            return self.name2node[str(name)].default_element

    def suggest_loguniform(self, name, low, high):
        if self.name2node[str(name)].status:
            return self.trial.suggest_loguniform(name, low, high)
        else:
            return self.name2node[str(name)].default_element

    def suggest_uniform(self, name, low, high):
        if self.name2node[str(name)].status:
            return self.trial.suggest_uniform(name, low, high)
        else:
            return self.name2node[str(name)].default_element



    def suggest_categorical(self, name, choices):
        new_list = []
        for element in choices:
            if self.name2node[str(name) + '##' + str(element)].status:
                new_list.append(element)

        if len(new_list) == 0:
            return copy.deepcopy(self.name2node[str(name)].default_element)

        return copy.deepcopy(categorical(self.trial, name, new_list))