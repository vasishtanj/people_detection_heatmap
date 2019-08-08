"""Store Aisle Monitor"""

"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
from argparse import ArgumentParser
import pathlib
import cv2
import numpy as np

from inference import Network

# Weightage/ratio to merge (for Heatmap) original frame and colorMap frame(sum of both should be 1)
INITIAL_FRAME_WEIGHTAGE = 0.65
COLORMAP_FRAME_WEIGHTAGE = 0.35

# Weightage/ratio to merge (for integrated output) people count frame and colorMap frame(sum of both should be 1)
P_COUNT_FRAME_WEIGHTAGE = 0.65
COLORMAP_FRAME_WEIGHTAGE_1 = 0.35

# Multiplication factor to compute time interval for uploading snapshots to the cloud
MULTIPLICATION_FACTOR = 5

# Azure Blob container name

# To get current working directory
CWD = os.getcwd()


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model",
                        help="Path to an .xml file with a trained model.",
                        required=True, type=str)
    parser.add_argument("-i", "--input",
                        help="Path to video file or image. Use 'cam' for "
                             "capturing video stream from camera",
                        required=True, type=str)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers. Absolute "
                             "path to a shared library with the kernels impl.",
                        type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; "
                             "CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. Application"
                             " will look for a suitable plugin for device "
                             "specified (CPU by default)", default="CPU", type=str)
    parser.add_argument("-pt", "--prob_threshold",
                        help="Probability threshold for detections filtering",
                        default=0.5, type=float)

    return parser


def main():
    args = build_argparser().parse_args()
    cap = cv2.VideoCapture(args.input)
    # Initialise the class
    infer_network = Network()
    # Load the network to IE plugin to get shape of input layer
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1, 0, args.cpu_extension)[1]

    print("To stop the execution press Esc button")
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 1
    accumulated_image = np.zeros((initial_h, initial_w), np.uint8)
    mog = cv2.createBackgroundSubtractorMOG2()
    ret, frame = cap.read()
    while cap.isOpened():
        ret, next_frame = cap.read()
        if not ret:
            break
        frame_count = frame_count + 1
        in_frame = cv2.resize(next_frame, (w, h))
        # Change data layout from HWC to CHW
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((n, c, h, w))

        # Start asynchronous inference for specified request.
        inf_start = time.time()
        infer_network.exec_net(0, in_frame)
        # Wait for the result
        infer_network.wait(0)
        det_time = time.time() - inf_start

        people_count = 0

        # Converting to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Remove the background
        fgbgmask = mog.apply(gray)
        # Thresholding the image
        thresh = 2
        max_value = 2
        threshold_image = cv2.threshold(fgbgmask, thresh, max_value,
                                                      cv2.THRESH_BINARY)[1]
        # Adding to the accumulated image
        accumulated_image = cv2.add(threshold_image, accumulated_image)
        colormap_image = cv2.applyColorMap(accumulated_image, cv2.COLORMAP_HOT)

        # Results of the output layer of the network
        res = infer_network.get_output(0)
        for obj in res[0][0]:
            # Draw only objects when probability more than specified threshold
            if obj[2] > args.prob_threshold:
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                class_id = int(obj[1])
                # Draw bounding box
                color = (min(class_id * 12.5, 255), min(class_id * 7, 255),
                              min(class_id * 5, 255))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                people_count = people_count + 1

        people_count_message = "People Count : " + str(people_count)
        inf_time_message = "Inference time: {:.3f} ms".format(det_time * 1000)
        cv2.putText(frame, inf_time_message, (15, 25), cv2.FONT_HERSHEY_COMPLEX, 1,
                         (255, 255, 255), 2)
        cv2.putText(frame, people_count_message, (15, 65), cv2.FONT_HERSHEY_COMPLEX, 1,
                         (255, 255, 255), 2)
        final_result_overlay = cv2.addWeighted(frame, P_COUNT_FRAME_WEIGHTAGE,
                                                    colormap_image,
                                                    COLORMAP_FRAME_WEIGHTAGE_1, 0)
        cv2.imshow("Detection Results", final_result_overlay)

        time_interval = MULTIPLICATION_FACTOR * fps

        frame = next_frame

        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    infer_network.clean()


if __name__ == '__main__':
    sys.exit(main() or 0)



