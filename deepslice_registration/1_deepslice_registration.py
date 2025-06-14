import DeepSlice
import pandas as pd
from glob import glob
import numpy as np

Model = DeepSlice.DSModel("mouse")

path = r"/home/harryc/github/AllenDownload/downloaded_data/*/*/10um"
folders = glob(path)

for folder in folders[-4:-3]:
    image_names = glob(folder + "/*")
    Model.predict(
        image_directory=None,
        image_list=image_names,
        ensemble=False,
        use_secondary_model=True,
    )
    image_folders = [i.split("/")[-3] for i in image_names]
    Model.predictions["Filenames"] = [
        f"{fol}/10um/{nam}"
        for fol, nam, in zip(image_folders, Model.predictions["Filenames"])
    ]
    Model.propagate_angles()
    Model.enforce_index_order()
    maxNumber = Model.predictions["nr"].max()
    if maxNumber < 250:
        section_thickness = 50
    elif maxNumber == 358:
        section_thickness = 35
    else:
        section_thickness = 25
    if "reference-2" in folder:
        section_thickness *= -1

    section_thickness = np.round(section_thickness)
    Model.enforce_index_spacing(section_thickness)
    specimen = folder.split("/")[-3]
    out_path = "/".join(folder.split("/")[:-2]) + "/" + specimen 

    Model.save_predictions(out_path)
    Model.save_predictions(f"registrations/{specimen}")
