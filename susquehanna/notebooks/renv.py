import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Load data
for filename in os.listdir("../data1999"):
    if filename.startswith("w"):
        globals()[f"{filename[:-4]}"] = np.loadtxt(f"../data1999/{filename}")
    elif filename == "salinity_min_flow_req.txt":
        globals()[f"{filename[:-4]}"] = np.loadtxt(
           os.path.join("../data1999", filename)
        )
    elif filename == "min_flow_req.txt":
        globals()[f"{filename[:-4]}"] = np.loadtxt(
            os.path.join("../data1999", filename)
        )

df = pd.read_csv("/Users/aas6791/PycharmProjects/MUSEH2O/susquehanna/notebooks/renv.csv", index_col=0)

renv = df.to_numpy()
min_flow_req = np.array(min_flow_req)
salinity_min_flow_req = np.array(salinity_min_flow_req)

# FERC only - areas between min_flow_req and each release in renv
areas_ferc = []

for release in renv:
    year_area = 0
    for item, req_flow in zip(release, min_flow_req):
        if item < req_flow:
            area = req_flow - item
        else:
            area = 0
        year_area += area
    areas_ferc.append(year_area)

# Salinity and FERC - areas between salinity_min_flow_req and each release in renv
areas_salinity = []

for release in renv:
    year_area = 0
    for item, sal_req_flow in zip(release, salinity_min_flow_req):
        if item < sal_req_flow:
            area = sal_req_flow - item
        else:
            area = 0
        year_area += area
    areas_salinity.append(year_area)

# Combine areas for normalization, concatenate lists with +
# norm is an instance of Normalize, which was applied to the whole range of possoble area values
all_areas = np.array(areas_ferc + areas_salinity)
norm = Normalize(vmin=all_areas.min(), vmax=all_areas.max())
cmap = plt.cm.jet

# Create the figure and axes
fig, axs = plt.subplots(2, 1, figsize=(16, 10))

# Plotting FERC minimum flow
for release, area in zip(renv, areas_ferc):
    # norm(area) then takes each individual area and applies the [0,1] range for color bar
    color = cmap(norm(area))
    axs[0].plot(release, c=color, linewidth=0.3, alpha=0.05)

axs[0].plot(min_flow_req, "black", ls="--", linewidth=1)
axs[0].set_ylabel("log(releases)")
axs[0].set_title("Dam Releases - FERC requirement only", loc="left", weight="bold")
axs[0].set_yscale("log")
axs[0].set_xlabel("day")

# Plotting salinity minimum flow
for release, area in zip(renv, areas_salinity):
    color = cmap(norm(area))
    axs[1].plot(release, c=color, linewidth=0.3, alpha=0.05)

axs[1].plot(salinity_min_flow_req, "black", ls="--", linewidth=1)
axs[1].set_ylabel("log(releases)")
axs[1].set_title("Dam Releases - Salinity and FERC requirements", loc="left", weight="bold")
axs[1].set_yscale("log")
axs[1].set_xlabel("day")

# Add the colorbar
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs, orientation="vertical", label="Amount of Violation")

plt.savefig(f"figs/combined_min_releases.svg")
#plt.show()

'''
#___________________________________________________________________________________________
# TODO: here is where you specify RBF to use here, itialize model, retrieve outputs...

#normallize for colorbar
norm = Normalize(vmin=min(areas_ferc), vmax=max(areas_ferc))
cmap = plt.cm.get_cmap('jet')

# Plotting FERC minimum flow
fig, ax = plt.subplots(figsize=(8, 2))

for release, area_ferc in zip(renv, areas_ferc):
    color = cmap(norm(area_ferc))
    ax.plot(release, c=color, linewidth=0.3, alpha=0.1)

# plot the other things
ax.plot(min_flow_req, "black", ls="--", linewidth=1)
ax.set_ylabel("log(releases)")
ax.set_title("Dam Releases - FERC requirements", loc="left", weight="bold")
ax.set_yscale("log")
ax.set_xlabel("day")

# Add the colorbar
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", label="Amount of Violation")

fig.tight_layout(pad=1.0)
#plt.savefig(f"figs/salinity_min_releases.svg")
plt.show()
#___________________________________________________________________________________________

# FERC and salinity plotting
# Concatenate lists of areas for normalization with +
all_areas = np.array(areas_ferc + areas_salinity)
norm = Normalize(vmin=min(all_areas), vmax=max(all_areas))
cmap = mpl.colormaps['jet']

# Plotting FERC minimum flow
fig, ax = plt.subplots()

for release, year_area in zip(renv, areas_ferc):
    color = cmap(norm(all_areas))
    ax.plot(release, c=color, linewidth=0.3, alpha=0.1)

ax.plot(min_flow_req, "black", ls="--", linewidth=1)
ax.set_ylabel("log(releases)")
ax.set_title("Dam Releases - FERC requirement only", loc="left", weight="bold")
ax.set_yscale("log")
ax.set_xlabel("day")

# Plotting Salinity minimum flow
fig, ax = plt.subplots()

for release, year_area in zip(renv, areas_salinity):
    color = cmap(norm(all_areas))
    ax.plot(release, c=color, linewidth=0.3, alpha=0.1)

ax.plot(salinity_min_flow_req, "black", ls="--", linewidth=1)
ax.set_ylabel("log(releases)")
ax.set_title("Dam Releases - Salinity and FERC requirements", loc="left", weight="bold")
ax.set_yscale("log")
ax.set_xlabel("day")

# Add the colorbar
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", label="Amount of Violation")

fig.tight_layout(pad=1.0)
#plt.savefig(f"figs/combined_min_releases.svg")
plt.show()
'''