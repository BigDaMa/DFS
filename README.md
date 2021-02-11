# Declarative Feature Selection
Responsible usage of Machine Learning (ML) systems in practice requires to enforce not only high prediction quality, but also to account for other constraints, such as fairness, privacy, or execution time. A simple way to address multiple user-specified constraints on ML systems is feature selection. Yet, applying feature selection to enforce user-specified constraints is challenging. Optimizing feature selection strategies with respect to multiple metrics is difficult to implement and has been underrepresented in previous experimental studies. Here, we propose Declarative Feature Selection (DFS) to simplify the design and validation of ML systems satisfying diverse user-specified constraints. We benchmark and evaluate a representative series of feature selection algorithms. From our extensive experimental results across 16 feature selection strategies, 19 datasets, and 3 classification models, we derive concrete suggestions on when to use which strategy and show that a meta-learning-driven optimizer can accurately predict the right strategy for an ML task at hand. These results demonstrate that feature selection can help to build ML systems that meet combinations of user-specified constraints, independent of the ML methods used. We believe that our empirical results and the proposed declarative feature selection will enable scientists and practitioners to better automate the design and validation of robust and trustworthy ML systems.

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

[Pareto-Optimal Results for the Test Set](../master/additional_charts/radar_charts_test_scores): We provide for all 19 datasets all pareto-optimal solution that declarative feature selection found in our benchmark.
Here is an example for such an Pareto front:
<img src="https://user-images.githubusercontent.com/5217389/82898629-f0423d00-9f59-11ea-9205-bb45367ac487.png" align="left" width="300" >





