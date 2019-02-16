# import the necessary packages
import argparse
import os
import time

import cv2
from nms import nms
import numpy as np

import utils
from decode import decode
from draw import drawPolygons, drawBoxes


def text_detection(image, east, min_confidence, width, height):
    image_name = image.split("/")[-1].split(".")[0]
    # load the input image and grab the image dimensions
    image = cv2.imread(image)
    orig = image.copy()
    (origHeight, origWidth) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (width, height)
    ratioWidth = origWidth / float(newW)
    ratioHeight = origHeight / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (imageHeight, imageWidth) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east)

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (imageWidth, imageHeight), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))


    # NMS on the the unrotated rects
    confidenceThreshold = min_confidence
    nmsThreshold = 0.2

    # decode the blob info
    (rects, confidences, baggage) = decode(scores, geometry, confidenceThreshold)

    offsets = []
    thetas = []
    for b in baggage:
        offsets.append(b['offset'])
        thetas.append(b['angle'])

    ##########################################################

    # functions = [nms.felzenszwalb.nms, nms.fast.nms, nms.malisiewicz.nms]
    function = nms.fast.nms
    polygons = utils.rects2polys(rects, thetas, offsets, ratioWidth, ratioHeight)

    print("[INFO] Running nms.polygons . . .")

    indicies = nms.polygons(polygons, confidences, nms_function=function, confidence_threshold=confidenceThreshold,
                             nsm_threshold=nmsThreshold)

    indicies = np.array(indicies).reshape(-1)

    drawpolys = np.array(polygons)[indicies]

    name = function.__module__.split('.')[-1].title()


    drawOn = orig.copy()
    drawPolygons(drawOn, drawpolys, ratioWidth, ratioHeight, (0, 255, 0), 2)

    cv2.imwrite("../output/" + image_name + "_output.png", drawOn)

def text_detection_command():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str,
        help="path to input image")
    ap.add_argument("-east", "--east", type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'frozen_east_text_detection.pb'),
        help="path to input EAST text detector")
    ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
        help="minimum probability required to inspect a region")
    ap.add_argument("-w", "--width", type=int, default=320,
        help="resized image width (should be multiple of 32)")
    ap.add_argument("-e", "--height", type=int, default=320,
        help="resized image height (should be multiple of 32)")
    args = vars(ap.parse_args())

    text_detection(image=args["image"], east=args["east"], min_confidence=args['min_confidence'], width=args["width"], height=args["height"], )


if __name__ == '__main__':
    text_detection_command()
