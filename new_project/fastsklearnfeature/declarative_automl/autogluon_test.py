import autogluon as ag

@ag.obj(
    x=ag.space.Real(-10, 10)
)
def objective(x):
    return (x - 2) ** 2

#study = optuna.create_study()
#study.optimize(objective, n_trials=100)