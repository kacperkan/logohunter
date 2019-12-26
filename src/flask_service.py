import argparse

import cv2
import flask
import flask_bootstrap
import numpy as np
import requests

import utils
from keras_yolo3.yolo import YOLO
from logos import detect_logo_with_loaded_image, match_logo_and_return_img
from similarity import compute_cutoffs
from utils import (
    load_extractor_model,
    load_features,
    model_flavor_from_name,
)

app = flask.Flask(__name__)
flask_bootstrap.Bootstrap(app)

model = None
logo = None
my_preprocess = None
features = None

sim_threshold = 0.95


@app.route("/")
@app.route("/index")
def index():
    return "Hello"


def get_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        "--yolo_model",
        type=str,
        dest="model_path",
        default="keras_yolo3/yolo_weights_logos.h5",
        help="path to YOLO model weight file",
    )
    parser.add_argument(
        "--anchors",
        type=str,
        dest="anchors_path",
        default="keras_yolo3/model_data/yolo_anchors.txt",
        help="path to YOLO anchors",
    )
    parser.add_argument(
        "--classes",
        type=str,
        dest="classes_path",
        default="data_classes.txt",
        help="path to YOLO class specifications",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        dest="score",
        default=0.1,
        help="YOLO object confidence threshold above which to show predictions",
    )
    parser.add_argument(
        "--gpu_num", type=int, default=1, help="Number of GPU to use"
    )
    parser.add_argument(  # good default choices: inception_logo_features_200_trunc2, vgg16_logo_features_128
        "--features",
        type=str,
        dest="features",
        default="inception_logo_features_200_trunc2.hdf5",
        help="path to LogosInTheWild logos features extracted by InceptionV3/VGG16",
    )

    flags = parser.parse_args()
    return flags


@app.route("/load_logo", methods=["POST"])
def load_logo():
    global logo
    r = flask.request
    np_img_bytes = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(np_img_bytes, cv2.IMREAD_COLOR)

    logo = img[..., ::-1]


@app.route("/process", methods=["POST"])
def process():
    r = flask.request
    np_img_bytes = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(np_img_bytes, cv2.IMREAD_COLOR)[..., ::-1]

    (
        img_input,
        feat_input,
        sim_cutoff,
        (bins, cdf_list),
    ) = compute_cutoffs(
        logo, (model, my_preprocess), features, sim_threshold
    )

    prediction, image = detect_logo_with_loaded_image(model, img)

    drawn_image = match_logo_and_return_img(
        image,
        prediction,
        (model, my_preprocess),
        (feat_input, sim_cutoff, bins, cdf_list, ["for_sure_logo"]),
    )

    return requests.post("/show_result", data=drawn_image)


def main():
    global model, my_preprocess, features

    flags = get_args()
    yolo = YOLO(
        **{
            "model_path": flags.model_path,
            "anchors_path": flags.anchors_path,
            "classes_path": flags.classes_path,
            "score": flags.score,
            "gpu_num": flags.gpu_num,
            "model_image_size": (416, 416),
        }
    )
    model = yolo

    model_name, flavor = model_flavor_from_name(flags.features)
    features, brand_map, input_shape = load_features(flags.features)

    model, preprocess_input, input_shape = load_extractor_model(
        model_name, flavor
    )
    my_preprocess = lambda x: preprocess_input(
        utils.pad_image(x, input_shape)
    )

    # cycle trough input images, look for logos and then match them against inputs
    text_out = ""
    for i, img_path in enumerate(FLAGS.input_images):
        text = img_path
        print(text)
        text_out += text

    app.run(port=5000, host="*")


if __name__ == "__main__":
    main()
