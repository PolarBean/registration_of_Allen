import DeepSlice
from glob import glob
import numpy as np
import os
import pandas as pd

path_to_images = "/home/harryc/github/AllenDownload/downloaded_data/"
alignment_files = glob("registrations/*.json")
combined_file = []
# get five sections evenly spaced for each file
for af in alignment_files:
    animal_name = os.path.basename(af).split(".")[0]
    df, target = DeepSlice.read_and_write.QuickNII_functions.read_QUINT_JSON(af)
    percentiles = [17, 35, 52, 69, 86]
    values = np.percentile(df["nr"], [percentiles], method="nearest")[0]
    df[df["nr"].isin(values)]
    df["Filenames"] = animal_name + "/" + df["Filenames"]
    combined_file.append(df)

df = pd.concat(combined_file)
DeepSlice.read_and_write.QuickNII_functions.write_QUINT_JSON(
    df, f"{path_to_images}/combined_deepslice", target=target, aligner="DeepSlice"
)
