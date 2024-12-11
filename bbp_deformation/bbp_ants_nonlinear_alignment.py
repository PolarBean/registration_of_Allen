#!/usr/bin/python
# -*- coding: latin-1 -*-



# =============================================================================================
# Librairies importation

import numpy as np
import nrrd
from scipy.ndimage import zoom
import sys
import re
import os
import cv2
import json
import shutil
import ants
import math

# =============================================================================================



# =============================================================================================
# Fonctions

def read_img(img_path = ""):
    """
    Reading a NPY or NRRD file
    @method read_img
    @param {String} img_path The path of the input image to be read (NPY or NRRD)
    @return {Array} A numpy array with values corresponding to the input image
    """

    ext = img_path.split(".")[-1]

    if ext == "nrrd":
        print("NRRD")
        img, hd = nrrd.read(img_path)

    elif ext == "npy":
        print("NPY")
        try:
            img = np.load(img_path)
        except ValueError:
            img = np.load(img_path, allow_pickle=True)

    elif ext == "nii" or ext == "gz":
        print("NIFTI")
        nii_data = nib.load(img_path)
        img_temp = nii_data.get_fdata()
        img = np.nan_to_num(img_temp, nan = 0.0)

    elif ext == "png":
        print("PNG")
        img = imageio.open(img_path)
        img = np.array(img)

    elif ext == "jpg":
        print("JPG")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = np.array(img)

    elif ext == "raw":
        print("RAW")
        img = np.fromfile(img_path, dtype=np.float32)

    else:
        print("ERROR: img_path " + ext + "file format not supported")
        return

    return img



def write_img(img_path = "", out_array_img = []):
    """
    Writing a NPY or NRRD file
    @method write_img
    @param {String} img_path The path of the image to be written (NPY or NRRD)
    @param {Array} out_array_img The array to be written at img_path
    @return {None} Writes the out_array_img at img_path
    """

    ext = img_path.split(".")[-1]
    if ext == "nrrd":
        nrrd.write(img_path, out_array_img)
    elif ext == "npy":
        np.save(img_path, out_array_img)
    else:
        print("ERROR: outuput file format not supported")
        return

    print("Image successfully written at path", img_path)


def largest_dimensions(img1 = [], img2 = []):
    """
    Detrmining the largest 2D image dimensions between img1 and img2
    @method largest_dimensions
    @param {Array} img1 The img1 slice to consider
    @param {Array} img2 The img2 slice to consider
    @return {Array} xmax,ymax The largest dimsensions in 2D
    """

    # Getting image dimensions
    x1, y1 = img1.shape
    x2, y2 = img2.shape

    # Establising the largest dimensions
    xmax, ymax = 0, 0
    if x1 >= x2:
        xmax = x1
    elif x2 > x1:
        xmax = x2
    else:
        print("ERROR: wrong x image dimensions")
        return
    if y1 >= y2:
        ymax = y1
    elif y2 > y1:
        ymax = y2
    else:
        print("ERROR: wrong y image dimensions")
        return

    return xmax, ymax



def adapt_field_of_view(img1 = [], img2 = []):
    """
    Adapting the filed of wiew from a target mage to a ref image
    @method adapt_field_of_view
    @param {Array} img1 The img1 slice to consider
    @param {Array} img2 The img2 slice to consider
    @return {Array} targ_img_fov The target image adapted to the reference image field of view
    """

    # Getting initial image dimensions
    x1, y1 = img1.shape
    x2, y2 = img2.shape

    # Get the highest dimensions
    xmax, ymax = largest_dimensions(img1, img2)

    # Create the output images
    output_img1 = np.zeros((xmax, ymax), dtype = "float32")
    output_img2 = np.zeros((xmax, ymax), dtype = "float32")

    # Calculating the borders
    xbor1, ybord1 = xmax-x1, ymax-y1
    xbor2, ybord2 = xmax-x2, ymax-y2
    xleft_half_bord1, yleft_half_bord1 = xbor1//2, ybord1//2
    xright_half_bord1, yright_half_bord1 = xbor1 - xleft_half_bord1, ybord1 - yleft_half_bord1
    xleft_half_bord2, yleft_half_bord2 = xbor2//2, ybord2//2
    xright_half_bord2, yright_half_bord2 = xbor2 - xleft_half_bord2, ybord2 - yleft_half_bord2

    # Writing the input images in new dimension images
    output_img1[xleft_half_bord1:xmax-xright_half_bord1, yleft_half_bord1:ymax-yright_half_bord1] = img1
    output_img2[xleft_half_bord2:xmax-xright_half_bord2, yleft_half_bord2:ymax-yright_half_bord2] = img2

    return output_img1, output_img2


def generate_target_slice(ouv = [], atlas = [], shift = 0):
    """
    Generating the target slice in the atlas volume corresponding to DeepSlice input results
    @method generate_target_slice
    @param {Table} ouv The output results from DeepSlice
    @param {Array} atlas The atlas volume to consider for extracting the corresponding target slice
    @param {Integer} shift The shift value to consider from the caudal part of the brain given the 0 point along the rostro-caudal axis from the Allen Institute
    @return {Array} data_im The output corresponding image identified in the atlas volume given the DeepSlice parameters and the shift
    """

    width = None
    height = None
    ox, oy, oz, ux, uy, uz, vx, vy, vz = ouv
    oy += shift
    width = np.floor(math.hypot(ux,uy,uz)).astype(int) + 1
    height = np.floor(math.hypot(vx,vy,vz)).astype(int) + 1
    data = np.zeros((width, height), dtype=np.uint32).flatten()
    xdim, ydim, zdim = atlas.shape
    y_values = np.arange(height)
    x_values = np.arange(width)
    hx = ox + vx * (y_values / height)
    hy = oy + vy * (y_values / height)
    hz = oz + vz * (y_values / height)
    wx = ux * (x_values / width)
    wy = uy * (x_values / width)
    wz = uz * (x_values / width)
    lx = np.floor(hx[:, None] + wx).astype(int) 
    ly = np.floor(hy[:, None] + wy).astype(int) 
    lz = np.floor(hz[:, None] + wz).astype(int) 
    valid_indices = (0 <= lx) & (lx < xdim) & (0 <= ly) & (ly < ydim) & (0 <= lz) & (lz < zdim)
    valid_indices = valid_indices.flatten()
    lxf = lx.flatten()
    lyf = ly.flatten()
    lzf = lz.flatten()
    valid_lx = lxf[valid_indices]
    valid_ly = lyf[valid_indices]
    valid_lz = lzf[valid_indices]
    atlas_slice = atlas[valid_lx,valid_ly,valid_lz]
    data[valid_indices] = atlas_slice
    data_im = data.reshape((height, width))

    return data_im



# def one_ISH_dataset_nonlinear_warping(input_json_path = "", input_root_folder = "", input_reoriented_nissl_vol_path = "", output_folder = "", output_ref_nissl_slices_folder = "", transfo_type = ""):
def one_ISH_dataset_nonlinear_warping(input_json_folder = "", input_root_folder = "", input_reoriented_nissl_vol_path = "", output_folder = "", transfo_type = "", start_id = 0):
    """
    Warping a whole ISH dataset
    @method one_ISH_dataset_nonlinear_warping
    @param {String} input_json_folder The folder where to find all the json files
    @param {String} input_root_folder The path of the input root folder where images are located
    @param {String} input_reoriented_nissl_vol_path The path of input reoriented nissl stained reference volume
    @param {String} output_folder The path of the ouptut folder where to write the registered images
    @param {String} output_ref_nissl_slices_folder The path of the folder where to write the ref nissl slices extracted
    @param {String} transfo_type The type of transformation for registration
    @return {None} Writes the output registered slices in output_folder
    """

    # Listing all the json files
    json_files_list = [f for f in os.listdir(input_json_folder) if f.endswith("json")]
    json_files_list = sorted(json_files_list)
    nb_json_files = len(json_files_list)
    json_files_path = [os.path.join(input_json_folder, f) for f in json_files_list]
    temp_folder = "/gpfs/bbp.cscs.ch/project/proj137/scratch/piluso/atlalign_cache/"

    for i in range(start_id,nb_json_files):
        print("\n\nProcessing json file " + str(i+1) +"/" + str(nb_json_files))

        # Reading json file
        with open(json_files_path[i], 'r') as f:
            json_data = json.load(f)
 
        # Reading the input nissl file
        nissl_vol = read_img(input_reoriented_nissl_vol_path)
        
        # Getting the name and anchoring sections
        slices_data = json_data["slices"]
        filenames = [slice_data["filename"] for slice_data in slices_data]
        nb_slices = len(filenames)
        print("number of slices:", nb_slices)
        filenames = [filenames[k].replace("10um", "25um") for k in range(nb_slices)]
            # filenames = [filenames[k].replace("/10um_new", "").replace(".jpg", "_Affine.nrrd") for k in range(nb_slices)]
        anchorings = [slice_data["anchoring"] for slice_data in slices_data]
        # ISH_ID = int(filenames[0].split("/")[0])

        for j in range(nb_slices):

            # Target_slices identification and warping
            print("\nProcessing slice " + str(j+1) + "/" + str(nb_slices) + "   |   " + filenames[j])

            try:
                # Setting the output paths
                output_reg_path = os.path.join(output_folder, json_files_list[i].split(".")[0], filenames[j].split(".")[0].replace("/25um_new", "") + "_" + transfo_type + ".nrrd")
                # output_df_path = os.path.join(output_folder, json_files_list[i].split(".")[0], filenames[j].split(".")[0].replace("/25um_new", "") + "_" + transfo_type + "_df.npy")
                output_path_non_linear_df = output_reg_path.replace(".nrrd", "_nonLinearDf.nii.gz")
                
                # if os.path.exists(output_reg_path):
                if os.path.exists(output_path_non_linear_df):
                    print("Skip: File already exists")
                    pass
                else:
                    if os.path.exists(os.path.dirname(output_reg_path)):
                        pass
                    else:
                        # Creating the output folder
                        print("Directory not existant, creating it:", os.path.dirname(output_reg_path))
                        os.makedirs(os.path.dirname(output_reg_path), exist_ok=True)

                    # Target slice
                    slice_path = os.path.join(input_root_folder, json_files_list[i].split(".")[0], filenames[j])
                    # print("slice_path", slice_path)
                    if os.path.exists(slice_path):
                        targ_slice = read_img(slice_path)
                        print("Target slice read")
                        max_val = np.max(targ_slice)

                        # Dark field
                        targ_slice = max_val - targ_slice
                        print("Dark field applied")

                        # Reference slice
                        alignment = json_data['slices'][j]['anchoring']
                        ref_slice = generate_target_slice(alignment, nissl_vol, 14)

                        # if output_reg_path == "/gpfs/bbp.cscs.ch/project/proj137/scratch/piluso/ISH4harry/01_non_linear_warping/ISH/05-2788/71717630/71661738_s0248_Affine_SyN.nrrd":
                        #     output_nissl_path = output_reg_path.replace(".nrrd", "_nissl.nrrd")
                        #     write_img(output_nissl_path, ref_slice)

                        print("Nissl reference slice generated")

                        # # Resampling
                        # scale_factors = (z_scaling_factor, y_scaling_factor, x_scaling_factor)
                        # resampled_img = zoom(img, scale_factors)

                        # Adapting field-of-view
                        ref_slice, targ_slice = adapt_field_of_view(ref_slice, targ_slice)
                        print("Field of view adapted")

                        # Matching data types
                        ref_slice = ref_slice.astype("float32")
                        targ_slice = targ_slice.astype("float32")

                        # Non-linear warping
                    
                        if np.all(ref_slice == 0):
                            print("Dismissed: REF slice = 0")
                            pass
                        elif np.all(targ_slice == 0):
                            print("Dismissed: TARG slice = 0")
                            pass
                        else:

                            # # Nissl writing
                            # # output_ref_nissl_slice_path = os.path.join(output_ref_nissl_slices_folder,  filenames[i].split("/")[0] + "/nissl_" + filenames[i].split("/")[-1].split(".jpg")[0] + ".nrrd")
                            # output_ref_nissl_slice_path = output_reg_path.replace(".nrrd", "_nissl.nrrd")
                            # # output_ref_nissl_slice_path = os.path.join(output_ref_nissl_slices_folder,  filenames[i].split("/")[0] + "/label_" + filenames[i].split("/")[-1].split(".jpg")[0] + ".nrrd")
                            # # print("output_ref_nissl_slice_path", output_ref_nissl_slice_path)
                            # if os.path.exists(output_ref_nissl_slice_path):
                            #     print("Skip: Nissl file already exists")
                            #     pass
                            # else:
                            #     write_img(output_ref_nissl_slice_path, ref_slice)
                            #     # write_img("/home/piluso/data/11_DeepSlice/05_warped_files/00_Affine/targ_slice.nrrd", targ_slice)

                            # # Apply warping
                            # cache_folder = os.path.join(temp_folder, filenames[j].split("/")[-1].split(".")[0])
                            # if os.path.exists(cache_folder):
                            #     pass
                            # else:
                            #     os.makedirs(cache_folder)
                            # print("cache_folder", cache_folder)
                            # reg_image, df_mov2ref = registration_antspy(ref_slice, targ_slice, transfo_type, cache_folder)
                            
                            # Aligning directly using antspy
                            ref_slice = ants.image_clone(ants.from_numpy(ref_slice), pixeltype="float")
                            targ_slice = ants.image_clone(ants.from_numpy(targ_slice), pixeltype="float")
                            mytx = ants.registration(fixed=ref_slice , moving=targ_slice, type_of_transform="SyN")

                            # Saving affine matrix
                            output_path_affine = output_reg_path.replace(".nrrd", "_affineTransfo.mat")
                            shutil.copy(mytx["fwdtransforms"][1], output_path_affine)
                            print("Affine transformation saved")

                            # Saving the non linear deformation field
                            output_path_non_linear_df = output_reg_path.replace(".nrrd", "_nonLinearDf.nii.gz")
                            shutil.copy(mytx["fwdtransforms"][0], output_path_non_linear_df)
                            print("Non linear deformation file saved")
                            
                            # # Applying registration
                            # reg_image = ants.apply_transforms(fixed=ref_slice, moving=targ_slice,transformlist=mytx["fwdtransforms"])
                            # write_img(output_reg_path, reg_image.numpy())
                            # print("Warping applied")

                            # Writing the ouptut images
                            # write_img(output_reg_path, reg_image)
                            # write_img(output_df_path, df_mov2ref)
                    else:
                        print("Skip: target slice does not exist")
                        pass

            except TypeError:
                print("Skip: TYPE_ERROR")
                pass

    return

# =============================================================================================



# =============================================================================================
# Executions

# Command to system
input_json_folder = os.path.abspath(sys.argv[1])
input_root_folder = os.path.abspath(sys.argv[2])
input_reoriented_nissl_vol_path = os.path.abspath(sys.argv[3])
output_folder = os.path.abspath(sys.argv[4])
# output_ref_nissl_slices_folder = os.path.abspath(sys.argv[5])
transfo_type = sys.argv[5]
start_id = int(sys.argv[6])

print("Starting one_ISH_dataset_nonlinear_warping...")
print("input_json_folder:", input_json_folder)
print("input_root_folder:", input_root_folder)
print("input_reoriented_nissl_vol_path:", input_reoriented_nissl_vol_path)
print("output_folder:", output_folder)
# print("output_ref_nissl_slices_folder:", output_ref_nissl_slices_folder)
print("transfo_type:", transfo_type)
print("start_id:", start_id)

print("Processing...")
# one_ISH_dataset_nonlinear_warping(input_json_path, input_root_folder, input_reoriented_nissl_vol_path, output_folder, output_ref_nissl_slices_folder, transfo_type)
one_ISH_dataset_nonlinear_warping(input_json_folder, input_root_folder, input_reoriented_nissl_vol_path, output_folder, transfo_type, start_id)
print("Done: Slices from the ISH dataset successfully warped")

# =============================================================================================
