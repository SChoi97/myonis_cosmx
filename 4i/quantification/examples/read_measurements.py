"""Usage: python read_measurements.py /absolute/path/myonis_4i_data.h5ad"""
import sys
import anndata as ad
from fouri_quantification.access import get_nuclei, get_myotubes, get_structures

run = ad.read_h5ad(sys.argv[1])
nuclei = get_nuclei(run)
myotubes = get_myotubes(run)
print(nuclei.groupby(["cell_line", "repeat"]).size())
print(myotubes[["cell_line", "repeat", "n_nuclei"]].head())
if "sc35" in run.uns["structures"]:
    sc35 = get_structures(run, "sc35")
    print(sc35[["parent_nucleus_uid", "parent_myotube_uid"]].head())
print(run.uns["feature_dictionary"].head())
