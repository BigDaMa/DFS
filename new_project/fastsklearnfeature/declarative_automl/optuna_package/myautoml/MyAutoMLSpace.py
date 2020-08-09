from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import categorical
import copy

class MyAutoMLSpace:
    def __init__(self):
        self.parameters_used = {} #this has to be a tree
        self.parameter_default = {}
        self.trial = None

    def generate_cat(self, name, my_list, default_element):
        self.parameter_default[str(name)] = default_element
        for element in my_list:
            self.parameters_used[str(name) + '##' + str(element)] = True

        print(self.parameters_used)

    def sample_parameters(self, trial):
        #create dependency graph

        for k, v in self.parameters_used.items():
            self.parameters_used[k] = trial.suggest_categorical(str(k), [True, False])

    def generate_number(self, name, default_element):
        self.parameter_default[str(name)] = default_element
        self.parameters_used[str(name)] = True

    def suggest_int(self, name, low, high, step=1, log=False):
        if self.parameters_used[str(name)]:
            return self.trial.suggest_int(name, low, high, step=step, log=log)
        else:
            return self.parameter_default[str(name)]

    def suggest_loguniform(self, name, low, high):
        if self.parameters_used[str(name)]:
            return self.trial.suggest_loguniform(name, low, high)
        else:
            return self.parameter_default[str(name)]

    def suggest_uniform(self, name, low, high):
        if self.parameters_used[str(name)]:
            return self.trial.suggest_uniform(name, low, high)
        else:
            return self.parameter_default[str(name)]



    def suggest_categorical(self, name, choices):
        new_list = []
        for element in choices:
            if self.parameters_used[str(name) + '##' + str(element)]:
                new_list.append(element)

        if len(new_list) == 0:
            return copy.deepcopy(self.parameter_default[str(name)])

        return copy.deepcopy(categorical(self.trial, name, new_list))