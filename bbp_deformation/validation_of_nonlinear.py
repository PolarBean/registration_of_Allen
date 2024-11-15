import os
from glob import glob
import nrrd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import json
import re
from alignment_functions import read_ants_affine, read_nonlinear, apply_affine_to_image, apply_nonlinear_to_image


alignments_path = r'/mnt/e/AllenDataalignmentProj/correcting_strains_correct/'

deformation_data_root = r'/mnt/g/ISH_deformations/*/'
brains = glob(f"{alignments_path}/*.json")
for brain in tqdm(brains):
    brain = os.path.basename(brain)
    brain = brain[:-5]
    alignment_path = os.path.join(alignments_path, brain + '.json')
    image_base_path = rf"/mnt/g/AllenDataalignmentProj/resolutionPixelSizeMetadata/ISH/{brain}/"
    out_im_path = rf"warped/{brain}/"


    with open(alignment_path, 'r') as f:
        alignment = json.load(f)


    for slice in alignment['slices']:
        filename = slice['filename']
        filename = filename.replace('10um_new', '25um_new')
        base_filename = os.path.basename(filename)
        image_id, _ = base_filename.split('_')
        section_number = slice['nr']
        image_path = os.path.join(image_base_path, filename)
        image = cv2.imread(image_path)
        linear_path_search = rf"{deformation_data_root}/{brain}/*/{image_id}_s{str(section_number).zfill(4)}_SyN_affineTransfo.mat"
        linear_path = glob(linear_path_search)

        non_linear_path_search = rf"{deformation_data_root}/{brain}/*/{image_id}_s{str(section_number).zfill(4)}_SyN_nonLinearDf.nii.gz"
        non_linear_path = glob(non_linear_path_search)
        if len(linear_path)==0:
            print("no such alignment file: ", linear_path_search)
            continue
        else:
            linear_path = linear_path[0]
        if len(non_linear_path)==0:
            print("no such alignment file: ", non_linear_path_search)
            continue
        else:
            non_linear_path = non_linear_path[0]
        try:
            affine_matrix = read_ants_affine(linear_path)
        except:
            print(f"Failed to read {linear_path}")
            continue
        try:
            non_linear_data, non_linear_height, non_linear_width = read_nonlinear(non_linear_path)
        except:
            print(f"Failed to read {non_linear_path}")
            continue
        adjusted_image = apply_affine_to_image(
            image, 
            affine_matrix, (non_linear_height, non_linear_width), mode='edge')
        warped_image = apply_nonlinear_to_image(adjusted_image, non_linear_data)
        savedir = f"/mnt/g/seb_data_validation/{brain}/"
        os.makedirs(f"{savedir}/{'/'.join(filename.split('/')[0:-1])}", exist_ok=True)
        savepath = f"{savedir}/{filename}"
        slice['filename'] = filename
        cv2.imwrite(savepath, warped_image * 255)
        
    

    with open(f"{savedir}/{brain}.json", 'w') as out:
        json.dump(alignment, out)

