# Standard library imports
import argparse
import os
import math
import random
import json

# Third party imports
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing
import h5py
from unidecode import unidecode

def get_df_mediapipe(data, width, height):
    df = pd.DataFrame(data)  
    df['videoname'] = df['videoname'].apply(lambda x: x.strip().lower())
    df['class'] = df['videoname'].apply(lambda x: "_".join(x.split('_')[:-1]))
    df['number'] = df['videoname'].apply(lambda x: x.split('_')[-1])
    df['out_range?'] = (df['x']*width > width) | (df['y']*height > height)
    print(f"{df.shape=}")
    print(f"{df.videoname.nunique()=}")
    df_or_mediapipe = df.loc[(df['out_range?']==False) &
                            (df['class']!="???"), :].reset_index(drop=True)

    return df, df_or_mediapipe


def get_df_cocopose(dict_data_coco):
    df_cocopose = pd.DataFrame(dict_data_coco)  
    df_cocopose['videoname'] = df_cocopose['videoname'].apply(lambda x: x.strip().lower())
    df_cocopose['class'] = df_cocopose['videoname'].apply(lambda x: "_".join(x.split('_')[:-1]))
    df_cocopose['number'] = df_cocopose['videoname'].apply(lambda x: x.split('_')[-1])
    print(f"{df_cocopose.shape=}")
    print(f"{df_cocopose.videoname.nunique()=}")
    df_or_cocopose = df_cocopose.loc[(df_cocopose['outlier?']==False) &
                            (df_cocopose['class']!="???"), :].reset_index(drop=True)

    return df_cocopose, df_or_cocopose


def filter_landmarks(df_or, list_landmarks_mp, list_landmarks_coco_converted, use_coco):

    #for col_mp, col_coco in zip(list_landmarks_mp, list_landmarks_coco_converted):
    #    df_or.loc[(df_or.n_landmark==col_coco), "n_landmark"] = col_mp
    print(f"Use coco {use_coco}")

    df_flag_lm = df_or.groupby(['videoname', 'n_frame', 'n_landmark']).x.count().unstack()
    df_flag_lm_orig = df_flag_lm.copy()
    if use_coco:
        for col_mp, col_coco in zip(list_landmarks_mp, list_landmarks_coco_converted): 
            df_flag_lm[col_mp].fillna(df_flag_lm[col_coco], inplace=True)
            df_or.loc[(df_or.n_landmark==col_coco), "n_landmark"] = col_mp
        
    df_flag_lm["have_landmarks?"] = df_flag_lm[list_landmarks_mp].sum(1) == len(list_landmarks_mp)

    df_check1 = df_flag_lm.reset_index().groupby("videoname").agg({"n_frame": "nunique"}).rename(columns={"n_frame": "n_frames"})

    df_flag_lm_v = df_flag_lm[["have_landmarks?"]].unstack()
    df_flag_lm_v["sum_all_frames"] = df_flag_lm_v.sum(1)
    df_flag_lm_v = df_flag_lm_v.reset_index()
    df_flag_lm_v = df_flag_lm_v[["videoname", "sum_all_frames"]]
    df_flag_lm_v.columns = ["videoname", "sum_all_frames"]
    df_flag_lm_v = df_flag_lm_v.join(df_check1, on="videoname")
    df_flag_lm_v["all_frames?"] = df_flag_lm_v["sum_all_frames"] == df_flag_lm_v["n_frames"]
    df_flag_lm_v = df_flag_lm_v.set_index("videoname")
    
    df_or = df_or.join(df_flag_lm["have_landmarks?"], on=["videoname", "n_frame"])
    df_or = df_or.join(df_flag_lm_v["all_frames?"], on="videoname")

    # applying filters - landmarks
    print()
    print("Filter: list of landmarks")
    df_or = df_or.loc[df_or.n_landmark.isin(list_landmarks_mp)]
    df_or = df_or.drop_duplicates(subset=["videoname", "n_frame", "n_landmark"]).reset_index(drop=True)
    print(f"Shape {df_or.shape} - N classes", df_or["class"].nunique(), 
        " - Number of videos", df_or["videoname"].nunique())

    print()
    print("Filter: frames that have all landmarks")
    df_or = df_or.loc[df_or["have_landmarks?"]==True]
    print(f"Shape {df_or.shape} - N classes", df_or["class"].nunique(), 
        " - Number of videos", df_or["videoname"].nunique())

    print()
    print("Filter: videos which all frames have those landmarks")
    df_or = df_or.loc[df_or["all_frames?"]==True]
    print(f"Shape {df_or.shape} - N classes", df_or["class"].nunique(), 
        " - Number of videos", df_or["videoname"].nunique())

    return df_or, df_flag_lm_orig, df_flag_lm, df_flag_lm_v


def frame_completion(df_or, min_frames, porc_frame_completion):
    #FRAME COMPLETION
    df_fc = df_or.loc[(df_or.n_frames<min_frames) &
                (df_or.n_frames>=(1 - porc_frame_completion) * min_frames)]
    df_fc["n_frame_to_complete"] = min_frames - df_fc["n_frames"]

    df_fc["lower_value"] = df_fc.apply(lambda x: 0 + math.floor(x["n_frame_to_complete"]/2), axis=1)
    df_fc["upper_value"] = df_fc.apply(lambda x: (x["n_frames"] - 1) - (x["n_frame_to_complete"] - math.floor(x["n_frame_to_complete"]/2)), axis=1)


    df_fc.loc[(((df_fc.upper_value - df_fc.lower_value + 1) < df_fc.n_frames) & 
                ((df_fc.lower_value>df_fc.n_frame) | (df_fc.upper_value<df_fc.n_frame))) |
                (((df_fc.upper_value - df_fc.lower_value + 1) == df_fc.n_frames) & 
                (df_fc.lower_value==df_fc.n_frame)), "repeat?"] = True

    df_repeat = df_fc.loc[df_fc["repeat?"]==True].reset_index(drop=True)

    df_repeat.loc[(df_repeat.lower_value>df_repeat.n_frame), "n_frame"] = df_repeat["n_frame"] - df_repeat["lower_value"]
    df_repeat.loc[(df_repeat.upper_value<df_repeat.n_frame), "n_frame"] = df_repeat["n_frame"] + (df_repeat["n_frames"] - df_repeat["upper_value"] - 1)
    df_repeat.loc[(df_repeat.n_frame==0) &
                    (df_repeat.n_frame_to_complete==1), "n_frame"] = -1

    df_or = pd.concat([df_or, df_repeat[df_or.columns]]).sort_values(by=["videoname", "n_frame"],
                                                    ascending=True)
    df_or_new_nframes = df_or.groupby("videoname").agg({"n_frame":["min", "nunique"]}).reset_index()
    df_or_new_nframes.columns = ["videoname", "n_frame_min", "n_frame_nunique"]

    df_or = df_or.merge(df_or_new_nframes.rename(columns={"n_frame_nunique": "n_frames_new"}), how="left")
    df_or["n_frame"] = df_or["n_frame"] + df_or["n_frame_min"] * -1

    df_or["n_frames"] = df_or["n_frames_new"]
    df_or.drop(["n_frames_new", "n_frame_min"], axis=1, inplace=True)

    print()
    print("Filter: fill missing frames to reach the minimum")
    print(f"Shape {df_or.shape} - N classes", df_or["class"].nunique(), 
        " - Number of videos", df_or["videoname"].nunique())
    #FRAME COMPLETION

    return df_or


def filter_n_frames(df_or, min_frames):
    # at least min frames
    print()
    print("Filter: min number of frames")
    df_or = df_or.loc[df_or.n_frames>=min_frames]
    print(f"Shape {df_or.shape} - N classes", df_or["class"].nunique(), 
        " - Number of videos", df_or["videoname"].nunique())

    # subsampling exactly min_frames
    xd = df_or.groupby(["videoname", "n_frame"]).agg({"n_frames": "first"})
    xd['rate'] = xd['n_frames'].apply(lambda x: math.ceil(x/min_frames))
    xd = xd.reset_index()
    xd['valid_frame?'] = xd['n_frame'] % xd['rate'] == 0
    xd['missing_frames'] = min_frames - xd['n_frames'].apply(lambda x: math.ceil(x/math.ceil(x/min_frames))) 
    
    xd_valid = xd.loc[(xd['valid_frame?']==False) &
                        (xd.missing_frames>0)]
    xd_valid['row_number_video'] = xd_valid.groupby(['videoname'])['n_frame'].cumcount() + 1 

    xd_valid['upper_value'] = xd_valid.apply(lambda x: math.floor((x['n_frames'] - (min_frames - x['missing_frames']))/2) 
                                                + x['missing_frames'] - math.floor(x['missing_frames']/2), axis=1)
    xd_valid['lower_value'] = xd_valid.apply(lambda x: math.floor((x['n_frames'] - (min_frames - x['missing_frames']))/2) 
                                                    - math.floor(x['missing_frames']/2), axis=1)

    xd_valid.loc[((xd_valid['lower_value']<xd_valid['upper_value']) & 
                (xd_valid['row_number_video']>xd_valid['lower_value']) & 
                (xd_valid['row_number_video']<=xd_valid['upper_value']))|
                ((xd_valid['lower_value']==xd_valid['upper_value']) &
                (xd_valid['row_number_video']==xd_valid['upper_value'])), 'valid_frame?'] = True

    xd = xd.merge(xd_valid[['videoname', 'n_frame', 'valid_frame?']].rename(columns={'valid_frame?': 'valid_frame2?'}),
                    how='left', on=['videoname', 'n_frame'])
    xd.loc[(~xd['valid_frame?']) & 
            (xd['valid_frame2?']), 'valid_frame?'] = xd['valid_frame2?']

    df_or = df_or.join(xd.set_index(['videoname', 'n_frame'])["valid_frame?"], on=['videoname', 'n_frame'])

    print()
    print(f"Filter: {min_frames} frames for each video")
    df_or = df_or.loc[df_or["valid_frame?"]].reset_index(drop=True)
    print(f"Shape {df_or.shape} - N classes", df_or["class"].nunique(), 
        " - Number of videos", df_or["videoname"].nunique())

    return df_or


def filter_n_instances(df_or, min_instances):
    df_or_class = df_or.groupby("class").agg({"videoname": "nunique"}).rename(columns={"videoname": "n_instances"})
    df_or = df_or.join(df_or_class, on="class")

    print()
    print(f"Filter: classes of at least {min_instances} instances")
    df_or = df_or.loc[df_or.n_instances>=min_instances].reset_index(drop=True)
    df_or = df_or.loc[df_or["class"]!="NNN"].reset_index(drop=True)
    print(f"Shape {df_or.shape} - N classes", df_or["class"].nunique(), 
        " - Number of videos", df_or["videoname"].nunique())

    return df_or

def undersampling_exact_n_instances(df_or, min_instances):
    ### UNDERSAMPLING TO HAVE THE SAME NUMBER OF INSTANCES OF EACH CLASS
    gg = df_or.groupby(["class", "videoname"]).n_instances.unique().reset_index()
    gg  = gg.groupby('class').apply(lambda x: x.sample(min_instances))
    gg["instance_to_use?"] = True
    df_or = df_or.join(gg.reset_index(drop=True).set_index(["class", "videoname"])["instance_to_use?"], on=["class", "videoname"])

    print()
    print(f"Filter: subsampling {min_instances} instances frames for each video")
    df_or = df_or.loc[df_or["instance_to_use?"]==True].reset_index(drop=True)
    print(f"Shape {df_or.shape} - N classes", df_or["class"].nunique(), 
        " - Number of videos", df_or["videoname"].nunique())
    ###
    return df_or






