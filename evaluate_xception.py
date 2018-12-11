import os
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from model2 import get_model, age_mae


def get_args():
    parser = argparse.ArgumentParser(description="APPA-REAL evaluation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file")
    args = parser.parse_args()
    return args

def error(predict, mean, standv):
    error = 1 - np.exp(-((predict-mean)*(predict-mean) / 2*standv*standv))
    return error

def main():
    args = get_args()
    depth = args.depth
    k = args.width
    weight_file = args.weight_file

    # load model and weights    
    img_size = 299
    batch_size = 32
    model = get_model(model_name="Xception")
    
    model.load_weights(weight_file)
    dataset_root = Path(__file__).parent.joinpath("appa-real", "appa-real-release")
    validation_image_dir = dataset_root.joinpath("test")
    gt_valid_path = dataset_root.joinpath("gt_avg_test.csv")
    image_paths = list(validation_image_dir.glob("*_face.jpg"))

    faces = np.empty((batch_size, img_size, img_size, 3))
    ages = []
    image_names = []

    for i, image_path in tqdm(enumerate(image_paths)):
        faces[i % batch_size] = cv2.resize(cv2.imread(str(image_path), 1), (img_size, img_size))
        image_names.append(image_path.name[:-9])

        if (i + 1) % batch_size == 0 or i == len(image_paths) - 1:
            results = model.predict(faces)
            ages_out = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages_out).flatten()
            ages += list(predicted_ages)
            
    print(len(ages))
    print(len(image_names))
    name2age = {image_names[i]: ages[i] for i in range(len(image_names))}
    df = pd.read_csv(str(gt_valid_path))
    appa_abs_error = 0.0
    real_abs_error = 0.0
    epsilon_error = 0.0
    count1 = 0
    count2 = 0
    iter = 0

    for i, row in df.iterrows():
        #iter += 1
        difference1 = name2age[row.file_name] - row.apparent_age_avg
        difference2 = name2age[row.file_name] - row.real_age
        appa_abs_error += abs(difference1)
        real_abs_error += abs(difference2)
        epsilon_error += error(name2age[row.file_name], row.apparent_age_avg, 0.3)
        ''''if int(difference1) == 0:
            count1 += 1
        if int(difference2) == 0:
            count2 += 1
        if iter < 5:
            print("Predicted age: {}".format(name2age[row.file_name]))'''
    print("MAE Apparent: {}".format(appa_abs_error / len(image_names)))
    print("MAE Real: {}".format(real_abs_error / len(image_names)))
    print("epsilon-error: {}".format(epsilon_error / len(image_names)))
    #print("Apparent age accuracy: {}".format(count1 / iter))
    #print("Real age accuracy: {}".format(count2 / iter))
    #print("len(image_names): {}".format(len(image_names)))
    #print("iter: {}".format(iter))

if __name__ == '__main__':
    main()