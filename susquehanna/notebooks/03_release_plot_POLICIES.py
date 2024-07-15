#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from platypus import Problem, EpsNSGAII, Real, ProcessPoolEvaluator
import csv
import logging
from itertools import chain

logging.basicConfig(level=logging.INFO)
plt.rcParams["figure.figsize"] = [12, 8]
# sys.path.append('..')

sys.path.append(os.path.abspath(".."))
from susquehanna_model import SusquehannaModel
import rbf_functions

rbfs =[
    rbf_functions.squared_exponential_rbf,
    rbf_functions.original_rbf,
    rbf_functions.inverse_quadratic_rbf,
    rbf_functions.inverse_multiquadric_rbf,
    rbf_functions.exponential_rbf,
    rbf_functions.matern32_rbf,
    rbf_functions.matern52_rbf,
]

pareto_sets = {}
for entry in rbfs:
    name = entry.__name__
    output_dir = "./refsets/"
    results = pd.read_csv(output_dir + name + "_refset.csv")
    results["environment"] = 1 - results["environment"]

    pareto_sets[name] = results

rbfs = [
    rbf_functions.squared_exponential_rbf,
    rbf_functions.original_rbf,
    rbf_functions.inverse_quadratic_rbf,
    rbf_functions.inverse_multiquadric_rbf,
    rbf_functions.exponential_rbf,
    rbf_functions.matern32_rbf,
    rbf_functions.matern52_rbf,
]

reference_sets = {}
for entry in rbfs:
    name = entry.__name__
    reference_sets[name] = pd.read_csv(
        os.path.join("./refsets", f"{name}_refset_with_variables.csv")
    )

for filename in os.listdir("../data1999"):
    if filename.startswith("w"):
        globals()[f"{filename[:-4]}"] = np.loadtxt(f"../data1999/{filename}")
    #elif filename == "salinity_min_flow_req.txt":
    elif filename == "min_flow_req.txt":
        globals()[f"{filename[:-4]}"] = np.loadtxt(
            os.path.join("../data1999", filename)
        )

# choose which RBF to run, don't want to print all 7
entry = rbfs[1]
reference_set = reference_sets[entry.__name__]

# set up the RBF network
n_inputs = 2  # (time, storage of Conowingo)
n_outputs = 4  # Atomic, Baltimore, Chester, Downstream:- (hydropower, environmental)
n_rbfs = n_inputs + 2
rbf = rbf_functions.RBF(n_rbfs, n_inputs, n_outputs, rbf_function=entry)

# Initialize model
nobjs = 6
n_years = 1
susquehanna_river = SusquehannaModel(
    108.5, 505.0, 5, n_years, rbf
)  # l0, l0_MR, d0, years
# l0 = start level cono, l0_MR = start level muddy run, d0 = startday > friday = 5

susquehanna_river.set_log(True)

output = []
# iterate over solutions
for _, row in reference_set.iloc[0:10, 0:32].iterrows():
    output.append(susquehanna_river.evaluate(row))

level_CO, level_MR, ratom, rbalt, rches, renv = susquehanna_river.get_log()

# plotting the releases
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))

for release in renv:
    ax.plot(release, c='grey', linewidth=0.3, alpha=0.1)
    ax.plot(min_flow_req, "black", ls="--", linewidth=1)
    ax.plot(salinity_min_flow_req, "black", ls="--", linewidth=1)
    ax.set_ylabel("log(releases)")
    ax.set_title("(d) Environmental requirements", loc="left", weight="bold")
    ax.set_yscale("log")
    ax.set_xlabel("day")
    

fig.tight_layout(pad=1.0)
#plt.savefig(f"figs/{name}/{name}_releases.jpg")
plt.show()




