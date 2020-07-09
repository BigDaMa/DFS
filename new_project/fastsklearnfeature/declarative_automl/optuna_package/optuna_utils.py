def categorical(trial, name, categories):
    list_id = trial.suggest_categorical(name, list(range(len(categories))))
    return categories[list_id]