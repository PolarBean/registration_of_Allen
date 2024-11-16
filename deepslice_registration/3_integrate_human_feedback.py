import DeepSlice
from glob import glob
import os

path_to_images = "/home/harryc/github/AllenDownload/downloaded_data/"
adjusted_file = rf"{path_to_images}/combined_deepslice_human_adjustments.json"
alignment_files = glob("registrations/*.json")
adjusted_df, t = DeepSlice.read_and_write.QuickNII_functions.read_QUINT_JSON(
    adjusted_file
)
adjusted_df_split = adjusted_df["Filenames"].str.split("/", expand=True).iloc[:, 1:]
adjusted_df["Filenames"] = adjusted_df_split.agg("/".join, axis=1)
adjusted_df["bad_section"] = False
# get five sections evenly spaced for each file
for af in alignment_files:
    animal_name = os.path.basename(af).split(".")[0]
    df, target = DeepSlice.read_and_write.QuickNII_functions.read_QUINT_JSON(af)
    df["bad_section"] = True
    df_mask = df["Filenames"].isin(adjusted_df["Filenames"])
    adjusted_df_mask = adjusted_df["Filenames"].isin(df["Filenames"])
    df[df_mask] = adjusted_df[adjusted_df_mask].values
    df = DeepSlice.coord_post_processing.angle_methods.propagate_angles(
        df, method="mean", species="mouse"
    )
    maxNumber = df["nr"].max()
    if maxNumber < 250:
        section_thickness = 50
    elif maxNumber == 358:
        section_thickness = 35
    else:
        section_thickness = 25
    if "reference-2" in af:
        section_thickness *= -1
    df = DeepSlice.coord_post_processing.spacing_and_indexing.space_according_to_index(
        df, section_thickness=section_thickness, voxel_size=25
    )
    DeepSlice.read_and_write.QuickNII_functions.write_QUINT_JSON(
        df,
        f"{path_to_images}/{animal_name}/{animal_name}",
        target=target,
        aligner="DeepSlice",
    )

    DeepSlice.read_and_write.QuickNII_functions.write_QUINT_JSON(
        df,
        f"human_corrected_registrations/{animal_name}",
        target=target,
        aligner="DeepSlice",
    )
