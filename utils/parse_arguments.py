# Standard library imports
import argparse
import random
import os
import sys
import json


def parse_arguments_generate_dataset():
    ap = argparse.ArgumentParser(description='Using of media pipe in PSL')
    
    ap.add_argument('--inputPath', type=str, default="/data/shuaman/psl_gan/Data/Videos/Segmented_gestures/",
                    help='relative path of images input.')

    ap.add_argument('-wlf', '--withLineFeature', default=False, action='store_true',
                    help="use line features")

    ap.add_argument('-lhl', '--leftHandLandmarks', default=False, action='store_true',
                    help='Get left hand landmarks')

    ap.add_argument('-rhl', '--rightHandLandmarks', default=False, action='store_true',
                    help='Get right hand landmarks')

    ap.add_argument('-fl', '--faceLandmarks', default=False, action='store_true',
                    help='Get face landmarks')

    ap.add_argument('-mf', '--minframes', type=int, default=10,
                    help='Number of frames of each video')

    ap.add_argument('-mi', '--mininstances', type=int, default=4,
                    help='Number of instances of each class')

    ap.add_argument('-aej', '--addExtraJoint', default=False, action='store_true',
                    help='Add the joint number 33 (middle point of 12 and 11)')

    args = ap.parse_args()

    return args