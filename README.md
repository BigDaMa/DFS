# Declarative Feature Selection

## Using our system
To run the experiments, first, you need to set the paths in a configuration file with the name of your machine. Examples can be found here: ~/new_project/fastsklearnfeature/configuration/resources

We provide a small jupyter notebook as an example: [Example Notebook](../master/new_project/fastsklearnfeature/interactiveAutoML/new_bench/multiobjective/metalearning/openml_data/notebook/Tutorial.ipynb)

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


git clone https://github.com/BigDaMa/Complexity-Driven-Feature-Construction.git
cd Complexity-Driven-Feature-Construction/new_project
python -m pip install .
```

## Experiments






