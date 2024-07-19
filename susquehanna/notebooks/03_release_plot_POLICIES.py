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
sys.path.append(os.path.abspath(".."))
from susquehanna_model import SusquehannaModel
from notebooks import rbf_functions

rbfs =[
    rbf_functions.squared_exponential_rbf,
    rbf_functions.original_rbf,
    rbf_functions.inverse_quadratic_rbf,
    rbf_functions.inverse_multiquadric_rbf,
    rbf_functions.exponential_rbf,
    rbf_functions.matern32_rbf,
    rbf_functions.matern52_rbf,
]

'''
pareto_sets = {}
for entry in rbfs:
    name = entry.__name__
    output_dir = "./refsets/"
    results = pd.read_csv(output_dir + name + "_refset.csv")
    results["environment"] = 1 - results["environment"]

    pareto_sets[name] = results
    
rbfs =[
    rbf_functions.squared_exponential_rbf,
    rbf_functions.original_rbf,
    rbf_functions.inverse_quadratic_rbf,
    rbf_functions.inverse_multiquadric_rbf,
    rbf_functions.exponential_rbf,
    rbf_functions.matern32_rbf,
    rbf_functions.matern52_rbf,
]
'''
reference_sets = {}
for entry in rbfs:
    name = entry.__name__
    reference_sets[name] = pd.read_csv(
        os.path.join("./refsets", f"{name}_refset_with_variables.csv")
    )

for filename in os.listdir("../data1999"):
    if filename.startswith("w"):
        globals()[f"{filename[:-4]}"] = np.loadtxt(f"../data1999/{filename}")
    elif filename == "min_flow_req.txt":
        globals()[f"{filename[:-4]}"] = np.loadtxt(
            os.path.join("../data1999", filename)
        )
    elif filename == "salinity_min_flow_req.txt":
        globals()[f"{filename[:-4]}"] = np.loadtxt(
            os.path.join("../data1999", filename)
        )

# ___________________________________________________________________________________________________________
# plot one RBF release plot with salinity/FERC req lines

# choose which RBF to run, don't want to print all 7

alpha = 0.1
lw = 0.3

# Specify the first RBF entry to use
entry = rbfs[0]
name = entry.__name__
reference_set = reference_sets[name]

numberOfRBF = 6  # numberOfInput + 2
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
susquehanna_river.set_log(True)

# Evaluate the model for each row in the reference set
for _, row in reference_set.iloc[:, 0:32].iterrows():
    susquehanna_river.evaluate(row)

# Retrieve outputs including renv
level_CO, level_MR, ratom, rbalt, rches, renv = susquehanna_river.get_log()

# from susquehanna_model import create_path
if not os.path.exists(f"figs/{name}/releases"):
    os.makedirs(f"figs/{name}/releases")

# plotting the releases
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16, 6))

for release in renv:
    line1, = ax.plot(release, c='grey', linewidth=0.3, alpha=0.4, label = 'Flow')
    line2, = ax.plot(min_flow_req, "black", ls="--", linewidth=1, label = 'FERC flow requirement')
    line3, = ax.plot(salinity_min_flow_req, "blue", ls="--", linewidth=1, label = 'Salinity and FERC flow requirement')
    ax.set_ylabel("log(releases)")
    ax.set_title("Daily Dam Releases", loc="left", weight="bold")
    ax.set_yscale("log")
    ax.set_xlabel("day")
    ax.legend(handles=[line1, line2, line3], loc='lower left')

fig.tight_layout(pad=1.0)
plt.savefig(f"figs/{name}/{name}_2req_releases.svg")
plt.show()

# ___________________________________________________________________________________________________________


