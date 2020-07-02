Code for the NeurIPS 2018 paper "Hunting for Discriminatory Proxies in Linear Regression Models"


## Links to the paper
* [Conference version](https://papers.nips.cc/paper/7708-hunting-for-discriminatory-proxies-in-linear-regression-models)
* [arXiv version](https://arxiv.org/abs/1810.07155)

## How to run the code
You will need an installation of Python 3 with some commonly used data analysis packages, such as `numpy`, `scipy`, `sklearn`, and `pandas`. Python 2 may work, but the code has not been tested with Python 2.

In addition, this version of the code uses [Gurobi](https://www.gurobi.com), which is better at solving the exact optimization problem than `cvxopt` is. Gurobi is proprietary software, but free licenses are available for academic users.

To run the exact optimization problem, run
```
python main.py <dataset> -e <epsilon>
```
where dataset is either `ssl` (Strategic Subject List) or `cc` (Communities and Crimes), and epsilon is your desired association threshold.

To run the approximate optimization problem, simply add the `-a` flag to the above command.
