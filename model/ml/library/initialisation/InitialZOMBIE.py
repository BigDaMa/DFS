import numpy as np
from scipy.sparse import vstack


class InitialZOMBIE(): #n_records_for_each_class

    def __init__(self, x, y, random_seed=42):
        self.x = x
        self.y = y

        self.random_state = np.random.RandomState(seed=random_seed)




    def generate(self):

        #sample ids
        classes = np.unique(self.y)
        class_2_id = {}
        for c_i in range(len(classes)):
            class_2_id[classes[c_i]] = c_i


        training_ids = []

        ids = np.arange(self.y.shape[0])
        self.random_state.shuffle(ids)

        records_per_class = np.zeros(len(classes))

        for y_i in range(self.y.shape[0]):

            label_id = ids[y_i]
            class_label = self.y[label_id]

            training_ids.append([ids[y_i]])
            records_per_class[class_2_id[class_label]] = 1

            if np.sum(records_per_class) >= len(classes):
                break


        #contruct training
        X_train = None
        y_train = []
        for x_i in range(len(training_ids)):
            if X_train == None:
                X_train = self.x[x_i]
            else:
                X_train = vstack((X_train, self.x[x_i]))
            y_train.append(self.y[x_i])

        return (X_train, y_train)






