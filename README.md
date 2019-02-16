# Text Recognition on Products
Project for text recognition on images of industrial products using EAST text detector.

## Introduction
OpenCV’s EAST (Efficient and Accurate Scene Text Detection) text detector is a deep learning model, based on a novel architecture and training pattern. It is capable of:

- running at near real-time at 13 FPS on 720p images and
- obtains state-of-the-art text detection accuracy.

[Link to paper](https://arxiv.org/pdf/1704.03155.pdf)

## Installation
`pip install -r requirements.txt`

## Usage
- The script to run text detection is located in folder  `text_detection`

- To run the detector on a single image:

 `python text_detection.py --east frozen_east_text_detection.pb --image path_to_image`

- The output image will be put in the folder `output`

- For usages of all parameters

  `python text_detection.py --help`
