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

# Local imports
from utils.parse_arguments import parse_arguments_generate_dataset
from utils.utils_getdata import get_df_mediapipe, get_df_cocopose, filter_landmarks, frame_completion, filter_n_frames, filter_n_instances, undersampling_exact_n_instances

#all the videos were resized to this
WIDTH = HEIGHT = 220

WIDTH_COCO = HEIGHT_COCO = 256


#list of landmarks to get
# see ->  https://google.github.io/mediapipe/solutions/pose.html

N_POSE_LANDMARKS = 33
LIST_LANDMARKS_dict = {
                "27": [
                    0, 2, 5,
                    11, 12,
                    13, 14,
                    15, 16,
                    21, 22
                    ], #33 p 21 lh 21 rh
                "24": [0, 1, 2, 3, 4, 5, 6, 7,
                    8, 9, 10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20, 21, 22]
                }

N_LHAND_LANDMARKS = 21
LIST_LHAND_MEDIAPIPE_dict = {
                "27": [
                    5, 8, 9, 12, 13, 16, 17, 20
                    ],
                "24": []    
                }

N_RHAND_LANDMARKS = 21
LIST_RHAND_MEDIAPIPE_dict = {
                "27": [
                    5, 8, 9, 12, 13, 16, 17, 20
                    ],
                "24": []    
                }

LIST_LANDMARKS_COCO_dict = {
                "27": [
                    0, 1, 2, 5, 6, 7, 8, 9, 10, 95, 116,
                    96, 99, 100, 103, 104, 107, 108, 111, 
                    117, 120, 121, 124, 125, 128, 129, 132
                    ],
                "24": [0, 65, 1, 68,
                        62, 2, 59,
                        3, 4,
                        77, 71,
                        5, 6,
                        7, 8,
                        9, 10,
                        111, 132,
                        99, 120,
                        95, 116]
                }

FINAL_COLUMNS = ["videoname", "axis", "n_frame", "n_landmark", "coordinate"]

RAW_DATA_FILENAME = 'raw_data_mediapipe.json'

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class GenerateDataset:
    def __init__(self, 
                input_path, 
                output_path,
                lefthand_lm, 
                righthand_lm, 
                raw_dataset,
                raw_coco_dataset):

        self.input_path = input_path
        self.output_path = output_path
        self.raw_dataset = raw_dataset
        self.raw_coco_dataset = raw_coco_dataset
        self.lefthand_lm = lefthand_lm
        self.righthand_lm = righthand_lm

        self.mp_holistic, self.holistic, self.mp_drawing, self.drawing_spec = self.get_solution_mediapipe()

        self.folder_list = self.get_folder_list()

        self.list_X = []
        self.list_Y = []
        self.list_pos= []
        self.list_frames = []
        self.list_videoname = []


    def get_solution_mediapipe(self):
        print("Holistic Model")
        mp_holistic = mp.solutions.holistic


        holistic = mp_holistic.Holistic(min_detection_confidence=0.5,
                                                min_tracking_confidence=0.5)

        # Drawing
        mp_drawing = mp.solutions.drawing_utils
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        return mp_holistic, holistic, mp_drawing, drawing_spec


    def get_folder_list(self):
        # Folder list of videos's frames
        if os.path.isdir(self.input_path):
            folder_list = [file for file in os.listdir(self.input_path)
                                if os.path.isdir(self.input_path+file)]
            print("Is Directory")
        else:
            folder_list = [args.inputPath]
        print(folder_list)
        return folder_list


    def get_mediapipe_data(self):
        if self.raw_dataset is None:
        
            for video_folder_name in self.folder_list:
                video_folder_path = self.input_path + video_folder_name
                video_folder_list = [file for file in os.listdir(video_folder_path)]

                for video_file in video_folder_list:
                    self.process_video(video_file, video_folder_path)

            self.holistic.close()

            data = {
                    "videoname": self.list_videoname,
                    "n_frame": self.list_frames,
                    "n_landmark": self.list_pos,
                    "x": self.list_X,
                    "y": self.list_Y,
                }

            raw_datapath = os.path.join(self.output_path, RAW_DATA_FILENAME) 
            print(f"Saving raw data got by mediapipe in json file: {raw_datapath}")
            with open(raw_datapath, 'w') as outfile:
                json.dump(data, outfile)

        else:
            raw_datapath = os.path.join(self.output_path, self.raw_dataset) 
            print(f"Loading raw data got by mediapipe from json file: {raw_datapath}")
            with open(raw_datapath) as json_file:
                data = json.load(json_file)

            self.list_videoname = data["videoname"]
            self.list_frames = data["n_frame"]
            self.list_pos = data["n_landmark"]
            self.list_X = data["x"]
            self.list_Y = data["y"]

        return data

    def get_coco_data(self, max_landmark):

        print("Reading coco raw dataset")
        raw_datapath_coco = os.path.join(self.output_path, self.raw_coco_dataset) 
        with open(raw_datapath_coco) as json_file:
            print(raw_datapath_coco)
            data_cocopose = json.load(json_file)

        dict_data = {}

        list_videoname = []
        list_frames = []
        list_pos = []
        list_X = []
        list_Y = []
        list_score = []
        list_outlier = []

        for data_un in data_cocopose:
            for frame, frame_keypoints in enumerate(data_un["keypoints"]):
        #         print(frame_keypoints)
        #         sys.exit(0)
                for pos, (x, y, score) in enumerate(zip(frame_keypoints["x"], frame_keypoints["y"], frame_keypoints["scores"])):
                    list_videoname.append(data_un["label"] + "_" + str(data_un["id"]))
                    list_frames.append(frame)
                    list_pos.append(pos + max_landmark + 1)
                    list_X.append(x/WIDTH_COCO)
                    list_Y.append(y/HEIGHT_COCO)
                    list_score.append(score)
                    list_outlier.append(frame_keypoints["outlier"])
                    
        dict_data = {
            "videoname": list_videoname,
            "n_frame": list_frames,
            "n_landmark": list_pos,
            "x": list_X,
            "y": list_Y,
            #"score": list_score,
            "outlier?": list_outlier
        }

        return dict_data



    def create_dataset(self, min_frames=10, min_instances=4, 
                    use_extra_joint=False, porc_frame_completion=0.2, 
                    n_landmarks=27, use_coco=False):
        LIST_LANDMARKS = LIST_LANDMARKS_dict[str(n_landmarks)]
        LIST_LHAND_MEDIAPIPE = LIST_LHAND_MEDIAPIPE_dict[str(n_landmarks)]
        LIST_RHAND_MEDIAPIPE = LIST_RHAND_MEDIAPIPE_dict[str(n_landmarks)]
        LIST_LANDMARKS_COCO = LIST_LANDMARKS_COCO_dict[str(n_landmarks)]

        data = self.get_mediapipe_data()

        max_landmark = max(data["n_landmark"])

        list_landmarks_mp = LIST_LANDMARKS + [i + N_POSE_LANDMARKS for i in LIST_LHAND_MEDIAPIPE]
        list_landmarks_mp = list_landmarks_mp + [i + N_POSE_LANDMARKS + N_LHAND_LANDMARKS for i in LIST_RHAND_MEDIAPIPE]
        list_landmarks_coco_converted = [i + max_landmark + 1 for i in LIST_LANDMARKS_COCO]
        list_landmarks_total = list_landmarks_mp + list_landmarks_coco_converted

        dict_data_coco = self.get_coco_data(max_landmark)

        df_or = self.filter_data(data, dict_data_coco, list_landmarks_mp, list_landmarks_coco_converted, min_frames=min_frames, min_instances=min_instances, porc_frame_completion=porc_frame_completion, use_coco=use_coco)
        df_or = df_or.sort_values(by=["videoname"], ascending=True).reset_index(drop=True)

        #checking
        assert df_or.groupby(["videoname", "n_frame"]).n_landmark.nunique().nunique()==1 and df_or.groupby(["videoname", "n_frame"]).n_landmark.nunique().unique()[0]==len(list_landmarks_mp) , "Frames dont have the same number of landmarks"
        assert df_or.groupby("videoname").agg({"n_frame": "nunique"}).n_frame.nunique()==1 and df_or.groupby("videoname").agg({"n_frame": "nunique"}).n_frame.unique()[0]==min_frames, f"Videos were not subsampled to {min_frames} frames"
        assert df_or.groupby("class").agg({"videoname": "nunique"}).videoname.nunique()==1 and df_or.groupby("class").agg({"videoname": "nunique"}).videoname.unique()[0]==min_instances, f"Classes dont have the same number of instances ({min_instances})"

        # classes
        name_classes_array = df_or.groupby('videoname')["class"].first().values
        name_classes_array = [unidecode(x) for x in name_classes_array]
        videonames_array = df_or.groupby('videoname')["class"].first().index
        assert len(videonames_array) == len(name_classes_array), "There is a problem with classes"
        #classes
        
        #reshaping
        df_or = df_or.set_index(["videoname", "n_frame", "n_landmark"])[["x", "y"]].stack().reset_index()
        df_or.rename(columns={"level_3": "axis", 0: "coordinate"}, inplace=True)

        list_dfs = [df_or[FINAL_COLUMNS]]
        if use_extra_joint:
            list_landmarks_mp.append(max(list_landmarks_mp)+1)
            df_new_point = df_or.loc[df_or.n_landmark.isin([11, 12])].groupby(["videoname", "axis", "n_frame"]).agg({"coordinate": "mean"}).reset_index()
            df_new_point["n_landmark"] = max(list_landmarks_mp) + 1
            list_dfs = list_dfs + [df_new_point[FINAL_COLUMNS]]

        df_or = pd.concat(list_dfs).sort_values(by=FINAL_COLUMNS[:-1], ascending=True)

        assert len(df_or) % (2 * min_frames * len(list_landmarks_mp)) == 0, "This shape is not correct"

        data_array = df_or['coordinate'].values.reshape((-1, 2, min_frames, len(list_landmarks_mp)))

        str_extra_frames = "_extraframes" if porc_frame_completion>0 else ""
        filename = f"data{str_extra_frames}_{min_frames}_{min_instances}_{len(list_landmarks_mp)}.pk"
        path_filename = os.path.join(self.output_path, filename)

        print(f"Saving data in {filename} file")
        pickle_data = {
            "data": data_array,
            "name_labels": name_classes_array,
            "videonames": videonames_array
        }

        pickle.dump(pickle_data, open(path_filename, 'wb'))

        return

    
    def process_video(self, video_file, video_folder_path):
        print("processing " + video_file.split('.')[0])
        video_seg_folder_name = video_folder_path+'/'+video_file.split('.')[0]

        # Create a VideoCapture object
        cap = cv2.VideoCapture(video_folder_path+'/'+video_file)

        # Check if camera opened successfully
        if (cap.isOpened() is False):
            print("Unable to read camera feed", video_seg_folder_name)

        idx = 0
        ret, frame = cap.read()

        while ret is True:
            self.process_frame(frame, video_file, idx)
            ret, frame = cap.read()
            idx += 1

    
    def process_frame(self, frame, video_file, idx):
        # Convert the BGR image to RGB before processing.
        imageBGR = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw annotations on the image
        annotated_image = frame.copy()
        annotated_image.flags.writeable = True

        holisResults = self.holistic.process(imageBGR)

        # POSE
        for posi, data_point in enumerate(holisResults.pose_landmarks.landmark):
            self.list_videoname.append(video_file[:-4])
            self.list_frames.append(idx)
            self.list_X.append(data_point.x)
            self.list_Y.append(data_point.y)
            self.list_pos.append(posi)

        # Left hand
        if self.lefthand_lm:
            if(holisResults.left_hand_landmarks):
                print("Mediapipe got left hand landmarks")
                for posi, data_point in enumerate(holisResults.left_hand_landmarks.landmark, start=N_POSE_LANDMARKS):
                    self.list_videoname.append(video_file[:-4])
                    self.list_frames.append(idx)
                    self.list_X.append(data_point.x)
                    self.list_Y.append(data_point.y)
                    self.list_pos.append(posi)
            else:
                print("Mediapipe couldnt get left hand landmarks")
                for posi in range(N_POSE_LANDMARKS, N_POSE_LANDMARKS + N_LHAND_LANDMARKS):
                    self.list_videoname.append(video_file[:-4])
                    self.list_frames.append(idx)
                    self.list_X.append(2*WIDTH)
                    self.list_Y.append(2*HEIGHT)
                    self.list_pos.append(posi)

        # Right hand
        if self.righthand_lm:
            if(holisResults.right_hand_landmarks):
                print("Mediapipe got right hand landmarks")
                for posi, data_point in enumerate(holisResults.right_hand_landmarks.landmark, start=N_POSE_LANDMARKS+N_LHAND_LANDMARKS):
                    self.list_videoname.append(video_file[:-4])
                    self.list_frames.append(idx)
                    self.list_X.append(data_point.x)
                    self.list_Y.append(data_point.y)
                    self.list_pos.append(posi)
            else:
                print("Mediapipe couldnt get right hand landmarks")
                for posi in range(N_POSE_LANDMARKS+N_LHAND_LANDMARKS, N_POSE_LANDMARKS+N_LHAND_LANDMARKS + N_RHAND_LANDMARKS):
                    self.list_videoname.append(video_file[:-4])
                    self.list_frames.append(idx)
                    self.list_X.append(2*WIDTH)
                    self.list_Y.append(2*HEIGHT)
                    self.list_pos.append(posi)



    def filter_data(self, data, dict_data_coco, list_landmarks_mp, list_landmarks_coco_converted, min_frames=10, min_instances=4, porc_frame_completion=0.2, use_coco=False):
        
        #Reading data
        df, df_or_mediapipe = get_df_mediapipe(data, WIDTH, HEIGHT)
        df_cocopose, df_or_cocopose = get_df_cocopose(dict_data_coco)
        
        df_or = pd.concat([df_or_mediapipe[["videoname", "n_frame", "n_landmark",
                                "x", "y", "class", "number"]], 
                            df_or_cocopose[["videoname", "n_frame", "n_landmark",
                                        "x", "y", "class", "number"]]]).reset_index(drop=True)

        print()
        print("Original")
        print(f"Shape {df_or.shape} - N classes", df_or["class"].nunique(), 
            " - Number of videos", df_or["videoname"].nunique())

        #Filter landmarks
        df_or, df_flag_lm_orig, df_flag_lm, df_flag_lm_v = filter_landmarks(df_or, list_landmarks_mp, list_landmarks_coco_converted, use_coco)

        #Getting number of frames
        df_or_nframes = df_or.groupby("videoname").agg({"n_frame": "nunique"}).rename(columns={"n_frame": "n_frames"})
        df_or = df_or.join(df_or_nframes, on="videoname")

        #Frame completion
        if porc_frame_completion>0:
            df_or = frame_completion(df_or, min_frames, porc_frame_completion)

        #Filter exactly n frames
        df_or = filter_n_frames(df_or, min_frames)

        #Filter at least n instances
        df_or = filter_n_instances(df_or, min_instances)

        #Filter exactly n instances
        df_or = undersampling_exact_n_instances(df_or, min_instances)

        return df_or


if __name__ == "__main__":
    args = parse_arguments_generate_dataset()

    input_path = args.inputPath
    output_path = args.outputPath
    lefthand_lm = args.leftHandLandmarks
    righthand_lm = args.rightHandLandmarks
    min_frames = args.minframes
    min_instances = args.mininstances
    use_extra_joint = args.addExtraJoint
    porc_frame_completion = args.porcFrameComplet
    raw_dataset = args.rawDataset
    raw_coco_dataset = args.rawCocoDataset
    n_landmarks = args.nLandmarks
    use_coco = args.useCoco
 
    set_seed(12345)

    gds = GenerateDataset(input_path, output_path, lefthand_lm, righthand_lm, raw_dataset, raw_coco_dataset)
    gds.create_dataset(min_frames, min_instances, use_extra_joint, porc_frame_completion, n_landmarks, use_coco)