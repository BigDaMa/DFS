# Declarative Feature Selection
There is a huge body of research on feature selection strategies, which try to find representations that yield high model accuracy, low search time, or a compressed representation. Due to the lack of an independent benchmark, data scientists face a choice problem that is even more aggravated  with the rise of novel metrics, such as fairness, privacy, safety against adversarial examples, and restrictions on training/inference time.
In this paper, we benchmark and evaluate a representative series of feature selection algorithms in the context of declarative feature selection where the user specifies a set of constraints for the desired feature set and the feature selection strategy has to find one satisfying feature set.
From our extensive experimental results across 16 feature selection strategies, 20 datasets, and 6 constraint types, we derive concrete suggestions on when to use which strategy and explore whether a meta-learning-driven optimizer can accurately predict the right strategy for a machine learning task at hand.

## Using our system
To run the experiments, first, you need to set the paths in a configuration file with the name of your machine. Examples can be found here: ~/new_project/fastsklearnfeature/configuration/resources

We provide a small jupyter notebook as an example: [open in nbviewer](https://nbviewer.jupyter.org/github/BigDaMa/DFS/blob/master/new_project/fastsklearnfeature/interactiveAutoML/new_bench/multiobjective/metalearning/openml_data/notebook/Tutorial-Adult.ipynb) / [open in Github](../master/new_project/fastsklearnfeature/interactiveAutoML/new_bench/multiobjective/metalearning/openml_data/notebook/Tutorial-Adult.ipynb)

![Selection_999(155)](https://user-images.githubusercontent.com/5217389/82896838-e965fb00-9f56-11ea-817d-b7f8fd5f1216.png)


## Datasets
We provide the [datasets](https://drive.google.com/file/d/1Pg_n8lUGxkBmyiKIuc3LPPQm-wpWBq5u/view?usp=sharing) in an archive.

## Setup 
```
conda create -n myenv python=3.7
conda activate myenv

git clone https://github.com/jundongl/scikit-feature.git
cd scikit-feature
python setup.py install
cd ..

git clone https://github.com/FelixNeutatz/adversarial-robustness-toolbox.git
cd adversarial-robustness-toolbox
git checkout felix_version
python -m pip install .
cd ..


git clone https://github.com/BigDaMa/DFS.git
cd DFS/new_project
python -m pip install .
```

## Additional Evaluations
In addition to the charts provided in the paper, we provide additional evaluations:

1) [Dataset-specific Results](../master/additional_charts/Dataset-specific%20DFS%20Results.pdf): We report the coverage and the fraction of ML scenarios that a strategy finished the fastest across all 16 feature selection strategies and all 20 datasets.
2) [Scaling Experiments for Instances and Features across all Strategies](../master/additional_charts/scaling_experiments)
3) [Pareto-Optimal Results for the Test Set](../master/additional_charts/radar_charts_test_scores): We provide for all 20 datasets all pareto-optimal solution that declarative feature selection found in evaluating 1200 different constraint sets.
Here, are the pareto-optimal solutions for the Adult dataset:
<img src="https://user-images.githubusercontent.com/5217389/82898629-f0423d00-9f59-11ea-9205-bb45367ac487.png" align="left" width="300" >





