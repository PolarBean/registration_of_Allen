import DeepSlice 
import pandas as pd
from glob import glob
import numpy as np

Model = DeepSlice.DSModel('mouse')

path = r"/home/harryc/github/AllenDownload/downloaded_data/*/*/10um"
folders = glob(path)

for folder in folders:
    Model.predict(image_directory = folder, 
                  ensemble = False, 
                  use_secondary_model = True)
    Model.propagate_angles()
    Model.enforce_index_order()
    maxNumber = Model.predictions["nr"].max()
    if maxNumber < 250:
        section_thickness = 50
    elif maxNumber == 358:
        section_thickness = 35
    else:
        section_thickness = 25
    section_thickness = np.round(section_thickness)
    Model.enforce_index_spacing(section_thickness)
    specimen = folder.split('/')[-3]
    out_path = '/'.join( folder.split('/')[:-2] ) + '/' + specimen + '.json'
    Model.save_predictions(out_path)
    Model.save_predictions(
        f"registrations/{specimen}.json"
    )