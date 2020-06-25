Anomaly Detection Initial Experiments
=====================================

In this repository you can find the code developed for the initial experiments on Anomaly Detection. These experiments are framed within the project "IPSTERS - IPSentinel Terrestrial Enhanced Recognition System", funded by "Fundação para a Ciência e a Tecnologia". For full info please follow [this link](https://joaopfonseca.github.io/projects/ipsters/).

Project Organization
------------

    .
    ├── LICENSE
    ├── Makefile
    ├── README.md
    ├── data/
    ├── docs/
    ├── experiments/
    ├── model_search.pkl
    ├── requirements.txt
    ├── results.csv
    ├── setup.py
    ├── src
    │   ├── __init__.py
    │   ├── data
    │   ├── experiment
    │   │   ├── __init__.py
    │   │   ├── autoencoder_experiment.py
    │   │   ├── full_experiment.py
    │   │   ├── initial_experiment.py
    │   │   ├── oversampling_filter_experiment.py
    │   │   └── utils.py
    │   ├── models
    │   │   ├── AutoEncoder.py
    │   │   ├── __init__.py
    │   │   ├── data_selection.py
    │   │   └── oversampling.py
    │   ├── reporting
    │   │   ├── __init__.py
    │   │   └── reports.py
    │   └── visualization
    │       ├── __init__.py
    │       └── visualize.py
    ├── test_environment.py
    └── tox.ini

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
