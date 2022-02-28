# Standard library imports
import argparse
import os
import math
import random

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


#all the videos were resized to this
WIDTH = HEIGHT = 220

#list of landmarks to get
# see ->  https://google.github.io/mediapipe/solutions/pose.html
LIST_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7,
                 8, 9, 10, 11, 12, 13, 14,
                 15, 16, 17, 18, 19, 20, 21, 22]

FINAL_COLUMNS = ["videoname", "axis", "n_frame", "n_landmark", "coordinate"]

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class GenerateDataset:
    def __init__(self, 
                input_path, 
                with_lf, 
                lefthand_lm, 
                righthand_lm, 
                face_lm):

        self.input_path = input_path
        self.lefthand_lm = lefthand_lm
        self.righthand_lm = righthand_lm
        self.face_lm = face_lm

        self.mp_holistic, self.holistic, self.mp_drawing, self.drawing_spec = self.get_solution_mediapipe(with_lf)

        self.folder_list = self.get_folder_list()

        self.list_X = []
        self.list_Y = []
        self.list_pos= []
        self.list_frames = []
        self.list_videoname = []


    def get_solution_mediapipe(self, with_lf):
        print("Holistic Model")
        mp_holistic = mp.solutions.holistic

        if with_lf:
            print("   + with Line Feature")
            holistic = mp_holistic.Holistic(upper_body_only=True,
                                                min_detection_confidence=0.5,
                                                min_tracking_confidence=0.5)
        else:
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


    def create_dataset(self, min_frames=10, min_instances=4, use_extra_joint=False):
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

        df_or = self.filter_data(data, min_frames=min_frames, min_instances=min_instances)

        #checking
        assert df_or.groupby(["videoname", "n_frame"]).n_landmark.nunique().std()==0 , "Frames dont have the same number of landmarks"
        assert df_or.groupby("videoname").agg({"n_frame": "nunique"}).n_frame.nunique()==1 and df_or.groupby("videoname").agg({"n_frame": "nunique"}).n_frame.unique()[0]==min_frames, f"Videos were not subsampled to {min_frames} frames"
        assert df_or.groupby("class").agg({"videoname": "nunique"}).videoname.nunique()==1 and df_or.groupby("class").agg({"videoname": "nunique"}).videoname.unique()[0]==min_instances, f"Classes dont have the same number of instances ({min_instances})"

        # classes
        le = preprocessing.LabelEncoder()
        le.fit(df_or["class"])
        classes_array = le.transform(df_or.groupby('videoname')["class"].first().values)
        name_classes_array = df_or.groupby('videoname')["class"].first().values
        name_classes_array = [unidecode(x) for x in name_classes_array]

        assert len(classes_array) == len(name_classes_array), "There is a problem with label encoder"

        #reshaping
        df_or = df_or.set_index(["videoname", "n_frame", "n_landmark"])[["x", "y"]].stack().reset_index()
        df_or.rename(columns={"level_3": "axis", 0: "coordinate"}, inplace=True)

        list_dfs = [df_or[FINAL_COLUMNS]]
        if use_extra_joint:
            LIST_LANDMARKS.append(33)
            df_new_point = df_or.loc[df_or.n_landmark.isin([11, 12])].groupby(["videoname", "axis", "n_frame"]).agg({"coordinate": "mean"}).reset_index()
            df_new_point["n_landmark"] = 33
            list_dfs = list_dfs + [df_new_point[FINAL_COLUMNS]]

        df_or = pd.concat(list_dfs).sort_values(by=FINAL_COLUMNS[:-1], ascending=True)

        assert len(df_or) % (2 * min_frames * len(LIST_LANDMARKS)) == 0, "This shape is not correct"

        data_array = df_or['coordinate'].values.reshape((-1, 2, min_frames, len(LIST_LANDMARKS)))

        filename = f"data_{min_frames}_{min_instances}_{len(LIST_LANDMARKS)}.pk"
        

        print(f"Saving data in {filename} file")
        pickle_data = {
            "data": data_array,
            "labels": classes_array,
            "name_labels": name_classes_array,
            "label_encoder": le
        }

        pickle.dump(pickle_data, open(filename, 'wb'))

        """
        print("Saving h5 files")
        h5_file = h5py.File(f"data_{min_frames}_{min_instances}_{len(LIST_LANDMARKS)}.h5", 'w')

        h5_file["data"] = data_array
        h5_file["labels"] = classes_array
        h5_file["name_labels"] = name_classes_array

        h5_file.close()
        """    
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
                for posi, data_point in enumerate(holisResults.left_hand_landmarks.landmark):
                    self.list_videoname.append(video_file[:-4])
                    self.list_frames.append(idx)
                    self.list_X.append(data_point.x)
                    self.list_Y.append(data_point.y)
                    self.list_pos.append(posi)
            else:
                print("Mediapipe couldnt get left hand landmarks")

        # Right hand
        if self.righthand_lm:
            if(holisResults.right_hand_landmarks):
                for posi, data_point in enumerate(holisResults.right_hand_landmarks.landmark):
                    self.list_videoname.append(video_file[:-4])
                    self.list_frames.append(idx)
                    self.list_X.append(data_point.x)
                    self.list_Y.append(data_point.y)
                    self.list_pos.append(posi)
            else:
                print("Mediapipe couldnt get right hand landmarks")

        # Face mesh
        if self.face_lm:
            if(holisResults.face_landmarks):
                for posi, data_point in enumerate(holisResults.face_landmarks.landmark):
                    self.list_videoname.append(video_file[:-4])
                    self.list_frames.append(idx)
                    self.list_X.append(data_point.x)
                    self.list_Y.append(data_point.y)
                    self.list_pos.append(posi)
            else:
                print("Mediapipe couldnt get face landmarks")

    def filter_data(self, data, min_frames=10, min_instances=4):
        df = pd.DataFrame(data)  

        df['class'] = df['videoname'].apply(lambda x: x.split('_')[0])
        df['number'] = df['videoname'].apply(lambda x: x.split('_')[1])
        df['out_range?'] = (df['x']*WIDTH > WIDTH) | (df['y']*HEIGHT > HEIGHT)

        df_or = df.loc[df['out_range?']==False, :].reset_index(drop=True)

        df_flag_lm = df_or.groupby(['videoname', 'n_frame', 'n_landmark']).x.count().unstack()
        df_flag_lm["have_landmarks?"] = df_flag_lm[LIST_LANDMARKS].sum(1) == len(LIST_LANDMARKS)

        df_check1 = df_flag_lm[df_flag_lm["have_landmarks?"]==True].reset_index().groupby("videoname").agg({"n_frame": ["sum", "max"]})
        df_check1.columns = [ x[0] + "_" + x[1] for x in df_check1.columns]
        df_check1["all_frames?"] = df_check1["n_frame_sum"] == df_check1["n_frame_max"]*(df_check1["n_frame_max"]+1)/2

        df_or = df_or.join(df_flag_lm["have_landmarks?"], on=["videoname", "n_frame"])
        df_or = df_or.join(df_check1["all_frames?"], on="videoname")

        # applying filters - landmarks
        print()
        print("Original")
        print(f"Shape {df_or.shape} - N classes", df_or["class"].nunique(), 
            " - Number of videos", df_or["videoname"].nunique())

        print()
        print("Filter: list of landmarks")
        df_or = df_or.loc[df_or.n_landmark.isin(LIST_LANDMARKS)]
        print(f"Shape {df_or.shape} - N classes", df_or["class"].nunique(), 
            " - Number of videos", df_or["videoname"].nunique())

        print()
        print("Filter: frames that have all landmarks")
        df_or = df_or.loc[df_or["have_landmarks?"]]
        print(f"Shape {df_or.shape} - N classes", df_or["class"].nunique(), 
            " - Number of videos", df_or["videoname"].nunique())

        print()
        print("Filter: videos which all frames have those landmarks")
        df_or = df_or.loc[df_or["all_frames?"]]
        print(f"Shape {df_or.shape} - N classes", df_or["class"].nunique(), 
            " - Number of videos", df_or["videoname"].nunique())

        df_or_nframes = df_or.groupby("videoname").agg({"n_frame": "nunique"}).rename(columns={"n_frame": "n_frames"})
        df_or = df_or.join(df_or_nframes, on="videoname")

        print()
        print("Filter: min number of frames")
        df_or = df_or.loc[df_or.n_frames>=min_frames]
        print(f"Shape {df_or.shape} - N classes", df_or["class"].nunique(), 
            " - Number of videos", df_or["videoname"].nunique())

        df_or_class = df_or.groupby("class").agg({"videoname": "nunique"}).rename(columns={"videoname": "n_instances"})
        df_or = df_or.join(df_or_class, on="class")

        print()
        print(f"Filter: classes of at least {min_instances} instances")
        df_or = df_or.loc[df_or.n_instances>=min_instances].reset_index(drop=True)
        df_or = df_or.loc[df_or["class"]!="NNN"].reset_index(drop=True)
        print(f"Shape {df_or.shape} - N classes", df_or["class"].nunique(), 
            " - Number of videos", df_or["videoname"].nunique())


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

        # subsampling min_frames
        xd = df_or.groupby(["videoname", "n_frame"]).agg({"n_frames": "first"})
        xd['rate'] = xd['n_frames'].apply(lambda x: math.ceil(x/min_frames))
        xd = xd.reset_index()
        xd['valid_frame?'] = xd['n_frame'] % xd['rate'] == 0
        xd['missing_frames'] = min_frames - xd['n_frames'].apply(lambda x: math.ceil(x/math.ceil(x/min_frames))) 
        
        xd_valid = xd.loc[(~xd['valid_frame?']) &
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


if __name__ == "__main__":
    args = parse_arguments_generate_dataset()

    input_path = args.inputPath
    with_lf = args.withLineFeature
    lefthand_lm = args.leftHandLandmarks
    righthand_lm = args.rightHandLandmarks
    face_lm = args.faceLandmarks
    min_frames = args.minframes
    min_instances = args.mininstances
    use_extra_joint = args.addExtraJoint

    set_seed(12345)

    gds = GenerateDataset(input_path, with_lf, lefthand_lm, righthand_lm, face_lm)
    gds.create_dataset(min_frames, min_instances, use_extra_joint)