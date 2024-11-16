"""
The left right orientation of Allen is the opposite of DeepSlice so we need to mirror the registrations
"""

import DeepSlice
from glob import glob
import numpy as np
import os

path_to_images = "/home/harryc/github/AllenDownload/downloaded_data/"
alignment_files = glob("human_corrected_registrations/*.json")

atlas_width = 456 / 2

columns = ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]

for af in alignment_files:
    filename = os.path.basename(af)
    animal_name = filename.split(".")[0]
    df, target = DeepSlice.read_and_write.QuickNII_functions.read_QUINT_JSON(af)
    """
    in retrospect atlas_width + 0.5 may be better, however this is compensated for 
    by the affine transform (bbp deform stage) which moves the section. Also QuickNII
    rounds the alignments to be the floor + 1 so this introduces more error than this .5 here
    """
    df["ox"] = atlas_width - (df["ox"] - atlas_width)
    df["ux"] *= -1
    df["vx"] *= -1
    DeepSlice.read_and_write.QuickNII_functions.write_QUINT_JSON(
        df,
        f"flipped_human_corrected_registrations/{animal_name}",
        target=target,
        aligner="DeepSlice",
    )
    DeepSlice.read_and_write.QuickNII_functions.write_QUINT_JSON(
        df,
        f"{path_to_images}/{animal_name}/{animal_name}_flipped",
        target=target,
        aligner="DeepSlice",
    )
