#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
from pywinauto.keyboard import send_keys
import pyttsx3
import time
import os
import threading
import pyautogui
from random import randint

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
import platform
import pygetwindow as gw
from tkinter import *

last_action_time = time.time()

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=CAMERA_DETECT_WIDTH)
    parser.add_argument("--height", help='cap height', type=int, default=CAMERA_DETECT_HEIGHT)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def send_back():
    print("N " + last_action_time)

def send_next():
    print("S " + last_action_time)

def get_focused_window_title():
    if platform.system() == "Windows":
        focused_window = gw.getActiveWindow()
        if focused_window is not None:
            return focused_window.title
        else:
            return "No window is currently focused"
    else:
        return "Not running on Windows"


def pptx_function(hand_sign_id):
    if hand_sign_id == 4:  # Skip
        pyautogui.press("down")
        print("PRINTED DOWN")
    if hand_sign_id == 5:  # Back
        pyautogui.press("up")
        print("PRINTED UP")


def main(app_mode):
    # Argument parsing #################################################################
    last_action_time = time.time()
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    current_line_points = []
    completed_lines = []

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0
    captcha1 = [[433, 335], [414, 273], [375, 213], [353, 164], [352, 123], [360, 224], [276, 224], [296, 237],
                [325, 238], [354, 268], [269, 261], [294, 270], [324, 274], [347, 311], [268, 299], [293, 305],
                [323, 308], [340, 351], [280, 337], [299, 336], [325, 339]]
    captcha2 = [[332, 347], [289, 336], [254, 296], [271, 258], [307, 249], [254, 233], [224, 197], [206, 173],
                [191, 153], [282, 216], [271, 165], [263, 135], [256, 111], [310, 216], [312, 170], [309, 140],
                [307, 115], [336, 228], [357, 196], [371, 173], [383, 153]]
    captcha3 = [[478, 151], [430, 166], [392, 189], [360, 207], [333, 216], [431, 237], [420, 286], [414, 315],
                [408, 339], [459, 247], [450, 300], [444, 332], [438, 357], [482, 249], [474, 298], [468, 328],
                [464, 351], [500, 245], [506, 282], [512, 304], [518, 322]]

    captcha_list = [captcha1, captcha2, captcha3]
    landmark_list1 = captcha_list[randint(0, len(captcha_list) - 1)]
    # landmark_list1 = captcha1
    landmark_list1_unitaire = pre_process_landmark(landmark_list1)
    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                cur_window = get_focused_window_title()
                print(cur_window)

                if hand_sign_id == 2:  # Point gesture
                    if app_mode == 0:
                        laser_point_history(landmark_list[8])
                        point_history.append([0, 0])
                    else:
                        point_history.append(landmark_list[8])

                if app_mode == 0 and (hand_sign_id == 4 or hand_sign_id == 5):  # pptx actions
                    print("aa")
                    print(cur_window)
                    print((time.time() - last_action_time) > 1)
                    print(cur_window.endswith("PowerPoint"))
                    print()
                    if (time.time() - last_action_time) > 1 and cur_window.endswith("PowerPoint"):
                        pptx_function(hand_sign_id)
                        last_action_time = time.time()

                    point_history.append([0, 0])

                if hand_sign_id == 2:
                    current_line_points.append([landmark_list[8][0], landmark_list[8][1]])
                elif hand_sign_id == 3 and current_line_points:
                    completed_lines.append(current_line_points.copy())
                    current_line_points.clear()

                    point_history.append([0, 0])

                # Eraser functionality
                if hand_sign_id == 6:
                    eraser_center = (landmark_list[8][0], landmark_list[8][1])
                    eraser_radius = 20
                    completed_lines = erase_lines(completed_lines, eraser_center, eraser_radius)

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks_and_eraser(debug_image, landmark_list, hand_sign_id)
                if app_mode == 1:
                    if close_enough(pre_processed_landmark_list, landmark_list1_unitaire):
                        is_human = True
                        # print("RAAAAAAAAAAAAH")
                    else:
                        # print("no")
                        is_human = False
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[most_common_fg_id[0][0]],
                        is_human,
                        1
                    )
                else:
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[most_common_fg_id[0][0]],
                        False,
                        app_mode
                    )


        else:
            point_history.append([0, 0])
        if app_mode ==2:
            debug_image = draw_all_lines(debug_image, current_line_points, completed_lines)

        if app_mode == 1:
            debug_image = draw_landmarks_and_eraser(debug_image, landmark_list1)
        debug_image = draw_info(debug_image, fps, mode, number, app_mode=app_mode)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # run_powerpoint()
    cap.release()
    cv.destroyAllWindows()


def draw_persistent_line(image, line_points):
    if len(line_points) > 1:
        for i in range(len(line_points) - 1):
            cv.line(image, (line_points[i][0], line_points[i][1]),
                    (line_points[i + 1][0], line_points[i + 1][1]), (0, 0, 0), 2)
    return image


def close_enough(list1, list2):
    cpt_false = 0
    cpt_true = 0
    for i in range(len(list1)):
        # print("there are the value that are being compared")
        # print(i," i",  "list 1 ", list1[i], " list 2 ", list2[i])
        if abs(list2[i]) < abs(list1[i] - (list1[i] * .35)) or abs(list2[i]) > abs(list1[i] + (list1[i] * .35)):
            cpt_false = cpt_false + 1
        else:
            cpt_true = cpt_true + 1
    total_cpt = cpt_true + cpt_false
    if (cpt_true / total_cpt) > .7:
        return True
    else:
        return False


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def calculate_movement_speed(point_history):
    speeds = []
    for i in range(1, len(point_history)):
        x1, y1 = point_history[i - 1]
        x2, y2 = point_history[i]
        speed = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  # Euclidean distance
        speeds.append(speed)
    return speeds


def is_rightward_movement(point_history):
    if len(point_history) < 2:
        return False
    return all(x2 > x1 for (x1, y1), (x2, y2) in zip(point_history[:-1], point_history[1:]))


def is_leftward_movement(point_history):
    if len(point_history) < 2:
        return False
    return all(x2 < x1 for (x1, y1), (x2, y2) in zip(point_history[:-1], point_history[1:]))


def is_consistent_movement(movement_speed, threshold=10, min_duration=5):
    consistent_speed = [speed for speed in movement_speed if speed > threshold]
    return len(consistent_speed) >= min_duration


def draw_lines(image, lines, color=(0, 0, 0), thickness=2):
    for line in lines:
        for i in range(1, len(line)):
            cv.line(image, (line[i-1][0], line[i-1][1]), (line[i][0], line[i][1]), color, thickness)
    return image

def draw_all_lines(image, current_line_points, completed_lines):
    image = draw_lines(image, completed_lines)
    if current_line_points:
        image = draw_lines(image, [current_line_points])
    return image


def draw_landmarks_and_eraser(image, landmark_point, hand_sign_id=0):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    if hand_sign_id == 6:
        eraser_center = (landmark_point[8][0], landmark_point[8][1])
        eraser_radius = 20  # Set the size of the eraser
        cv.circle(image, eraser_center, eraser_radius, (0, 0, 255), -1)  # Red circle

    return image

def erase_lines(lines, eraser_center, eraser_radius):
    new_lines = []
    for line in lines:
        new_line = []
        for point in line:
            if cv.norm(np.array(eraser_center) - np.array(point)) > eraser_radius:
                new_line.append(point)
            else:
                if len(new_line) > 1:
                    new_lines.append(new_line)
                new_line = []
        if new_line:
            new_lines.append(new_line)
    return new_lines

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text, thing, app_mode):
    print(app_mode)
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if app_mode != 1:
        print("si si ")
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        if finger_gesture_text != "":
            cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                       cv.LINE_AA)


    if thing:
        #print("we are in teh fukcing condition")
        cv.putText(image, "Congratulations you are indeed human", (10, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    prev_point = None
    for point in point_history:
        if point[0] != 0 and point[1] != 0:
            if prev_point is not None:
                cv.line(image, (prev_point[0], prev_point[1]), (point[0], point[1]), (0, 0, 0), 2)
            prev_point = point
    return image


def laser_point_history(point):
    if point[0] != 0 and point[1] != 0:
        pyautogui.moveTo((point[0] * SCREEN_WIDTH)/CAMERA_DETECT_WIDTH, (point[1] * SCREEN_HEIGHT)/CAMERA_DETECT_HEIGHT)


def draw_info(image, fps, mode, number, app_mode=0):
    if app_mode == 1:
        cv.putText(image, "VERIFY THAT YOU ARE A HUMAN", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)
        cv.putText(image, "Copy the gesture that is on the screen", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                   2, cv.LINE_AA)
    else:
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv.LINE_AA)


    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

def run_powerpoint():
    def open_presentation():
        presentation_path = r"TestPowerpoint.pptx"
        os.startfile(presentation_path)
        time.sleep(5)

        engine = pyttsx3.init()

    thread = threading.Thread(target=open_presentation)
    thread.start()
    main(0)
    mainMenu.draw_buttons(False)

def run_captcha():
    class Captcha_Menu:
        def __init__(self):
            self.btn_list = []
            self.root = Tk()
            self.root.title("ARE YOU HUMAN ??")
            self.root.geometry('500x400')

        def start(self):
            self.root.mainloop()


    main(1)

    mainMenu.draw_buttons(False)

def run_draw():
    main(2)

    mainMenu.draw_buttons(False)


class MainMenu:
    def __init__(self):
        self.btn_list = []
        self.root = Tk()
        self.root.title("ByteMasters")
        self.root.geometry('400x300')

    def start(self):
        self.draw_buttons()

        self.root.mainloop()

    def go_back(self):
        self.draw_buttons()

    def draw_buttons(self, show=True):
        if not show:
            for btn in self.btn_list:
                btn.place_forget()
            btn_back = Button(self.root, text='🔙 Back', command=self.go_back)
            btn_back.place(x=100, y=25)
        else:
            btn_pptx = Button(self.root, text='Present 😎', command=run_powerpoint)
            btn_pptx.place(x=100, y=25)

            btn_captcha = Button(self.root, text='Are you a robot? 🤖', command=run_captcha)
            btn_captcha.place(x=100, y=125)

            btn_draw = Button(self.root, text='Let\'s draw! 🎨', command=run_draw)
            btn_draw.place(x=100, y=225)

            self.btn_list = [btn_pptx, btn_captcha, btn_draw]



if __name__ == '__main__':
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080

    CAMERA_DETECT_WIDTH = 960
    CAMERA_DETECT_HEIGHT = 540

    is_human = False
    # run_powerpoint()
    # main()
    # app_mode -> 0: Present 1: Captcha 2: Draw



    mainMenu = MainMenu()
    mainMenu.start()

