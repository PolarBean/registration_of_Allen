import DeepSlice
import pandas as pd
from glob import glob
import numpy as np

Model = DeepSlice.DSModel("mouse")

path = r"/mnt/g/AllenDataalignmentProj/resolutionPixelSizeMetadata/ISH/*"
folders = glob(path)
specimens = np.unique([f.split('/')[-1] for f in folders])
template = "/mnt/g/AllenDataalignmentProj/resolutionPixelSizeMetadata/ISH/{}/*/10um_new/*.jpg"
for specimen in specimens:
    image_names = glob(template.format(specimen))
    Model.predict(
        image_directory=None,
        image_list=image_names,
        ensemble=False,
        use_secondary_model=True,
    )
    image_folders = np.unique([i.split("/")[-3] for i in image_names])
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
    if "reference-2" == specimen:
        section_thickness *= -1

    section_thickness = np.round(section_thickness)
    Model.enforce_index_spacing(section_thickness)
    Model.save_predictions(f"registrations/{specimen}")
