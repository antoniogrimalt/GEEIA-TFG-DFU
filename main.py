import tkinter
from tkinter import filedialog
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict
from sensor_msgs.msg import Image
import rosbag
import cv2
from cv_bridge import CvBridge
from retinaface import RetinaFace
import time
import json
import numpy as np
import pandas as pd
from std_msgs.msg import String

# TEMP
recognizable_persons_col = '[{"type":"PATIENT","appearances":[{"start":{"frame":126682,"landmarks":{"left_eye":[272,1]}},"end":{"frame":126923}}]},{"type":"PATIENT","appearances":[{"start":{"frame":126913,"landmarks":{"right_eye":[1279,330]}},"end":{"frame":127070}}]}]'

COLOR_DATA_TOPIC = "/device_0/sensor_1/Color_0/image/data"

# File types
BAG_FILE_TYPE = ("BAG File", "*.bag")
EXCEL_FILE_TYPE = ("Excel Workbook", "*.xlsx, *.xlsm, *.xlsb, *.xls")
JSON_FILE_TYPE = ("JSON File", "*.json")

FACIAL_AREA_MARGIN = 40

# Config
FACE_REUSE_LIMIT = 3
LANDMARK_DISTANCE_THRESHOLD = 50
MAX_CONFIDENCE_SCORE = 1.0000000000000000
IGNORE_STAFF = False  # Skip blurring the Staff
# Wound visibility
KEEP_COMPLETELY_OUT = False
KEEP_PARTIALLY_OUT = True
PARTIALLY_OUT_TOLERANCE = 1  # 1: LIGHTLY, 2: MODERATELY, 3: HEAVILY


def get_file_path(file_type: Tuple[str, str]) -> str:

    main_window = tkinter.Tk()
    # main_window.title("TFG")
    main_window.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(filetypes=[file_type])

    main_window.destroy()  # Close the main window
    return file_path


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    The IoU is a measure of the overlap between two bounding boxes. It is calculated
    as the area of the intersection divided by the area of the union of the boxes.

    Parameters:
    box1 (List[int]): Coordinates of the first bounding box in the format [x_min, y_min, x_max, y_max].
    box2 (List[int]): Coordinates of the second bounding box in the format [x_min, y_min, x_max, y_max].

    Returns:
    float: The IoU value between the two bounding boxes.
    """

    # Extract the coordinates for both bounding boxes
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the area of both bounding boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Determine the coordinates of the intersection rectangle
    inter_x_min = max(x1_min, x2_min)  # Leftmost x-coordinate
    inter_y_min = max(y1_min, y2_min)  # Topmost y-coordinate
    inter_x_max = min(x1_max, x2_max)  # Rightmost x-coordinate
    inter_y_max = min(y1_max, y2_max)  # Bottommost y-coordinate

    # Calculate the area of the intersection rectangle
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # Calculate the union area
    union_area = box1_area + box2_area - inter_area

    # Calculate the Intersection over Union (IoU) metric
    iou = inter_area / union_area

    return iou


def process_cv_image(
    cv_image: np.ndarray,
    current_segment: dict,
    current_persons: List[dict],
    frame_number,
) -> np.ndarray:

    image_height, image_width = cv_image.shape[:2]

    # Detect faces using RetinaFace
    detected_faces = RetinaFace.detect_faces(img_path=cv_image, threshold=0.1)

    assigned_person_ids = set()

    # If faces were detected
    if detected_faces is not None:
        # Iterate through them
        for face_code, detected_face in list(detected_faces.items()):
            matched_person_id = None
            max_iou = 0.0

            for person in current_persons:

                # Skip if this person's id is already assigned
                if person["id"] in assigned_person_ids:
                    continue

                # If this person's face data hasn't been updated yet
                if person["facial_area"] is None:
                    # Initialize variables to calculate average distance
                    total_distance = 0
                    valid_landmark_count = 0

                    #
                    for landmark, landmark_value in person["landmarks"].items():
                        if landmark_value is not None:
                            detected_point = np.array(
                                detected_face["landmarks"][landmark]
                            )
                            known_point = np.array(landmark_value)

                            distance = np.linalg.norm(detected_point - known_point)

                            total_distance += distance
                            valid_landmark_count += 1

                    if valid_landmark_count > 0:
                        average_distance = total_distance / valid_landmark_count
                        if average_distance < LANDMARK_DISTANCE_THRESHOLD:
                            matched_person_id = person["id"]
                            break
                else:
                    # Calculate IOU between the person's facial area and the detected facial area
                    iou = calculate_iou(
                        box1=person["facial_area"],
                        box2=detected_face["facial_area"],
                    )
                    if iou > max_iou:
                        max_iou = iou
                        matched_person_id = person["id"]

            if matched_person_id is not None:
                # Ensure the facial area touches the frame border if needed
                for landmark in detected_face["landmarks"].values():
                    x, y = landmark
                    if x < 0:
                        detected_face["facial_area"][0] = 0  # Replace x_min
                    elif x > image_width:
                        detected_face["facial_area"][2] = image_width  # Replace x_max
                    if y < 0:
                        detected_face["facial_area"][1] = 0  # Replace y_min
                    elif y > image_height:
                        detected_face["facial_area"][3] = image_height  # Replace y_max

                # Save the person's id in the face dict
                detected_face["person_id"] = matched_person_id

                # Add the matched person's id to the assigned set
                assigned_person_ids.add(matched_person_id)
            else:
                # print(
                #     f"Frame: {frame_number} - To be deleted: {detected_faces[face_code]}"
                # )
                # Remove the detected face if no good person match is found
                del detected_faces[face_code]

        # If not all detected faces were deleted
        if len(detected_faces) > 0:
            # print(f"Frame: {frame_number} - Remaining faces: {detected_faces}")

            # Update the face data of the persons present in the segment
            update_persons(
                current_persons=current_persons,
                detected_faces=detected_faces,
            )

    # Check for missing persons
    missing_person_ids = set(current_segment["headcount"]) - assigned_person_ids

    # If there are persons missing in the detection list
    if len(missing_person_ids) > 0:
        # If no faces were detected
        if detected_faces is None:
            # Initialize the detected_faces dict
            detected_faces = {}

        # Iterate through the missing persons to check if their previous facial data exists and can be used
        for index, person_id in enumerate(missing_person_ids, start=1):
            # If the reuse limit for the same face data hasn't been reached yet
            """
            Doesn't make sense, because the objective is to blur the face.
            A limit would mean that maybe there are going to be frames without blurring.
            Better to have a misaligned blur than none.
            """
            if current_persons[person_id]["face_data_usage_count"] < FACE_REUSE_LIMIT:
                """
                Not yet sure if this should be inside the following if condition.
                We are trying to not be more than X frames far from the original position.
                """
                current_persons[person_id]["face_data_usage_count"] += 1

                if current_persons[person_id]["facial_area"] is not None:
                    detected_faces[f"previous_{index}"] = {
                        "score": MAX_CONFIDENCE_SCORE,
                        "facial_area": current_persons[person_id]["facial_area"],
                        "landmarks": current_persons[person_id]["landmarks"],
                        "person_id": person_id,
                    }

    if len(detected_faces) > 0:
        # Blur each detected face
        for detected_face in detected_faces.values():
            x1, y1, x2, y2 = detected_face["facial_area"]

            # Blur the detected face region
            cv_image[y1:y2, x1:x2] = cv2.GaussianBlur(
                src=cv_image[y1:y2, x1:x2], ksize=(51, 51), sigmaX=30
            )

            # Print the person id in bold and white font
            person_id = detected_face["person_id"]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            color = (255, 255, 255)  # White color
            text_size, _ = cv2.getTextSize(str(person_id), font, font_scale, thickness)
            text_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_y = y1 + (y2 - y1 + text_size[1]) // 2
            cv2.putText(
                img=cv_image,
                text=f"{person_id}",
                org=(text_x, text_y),
                fontFace=font,
                fontScale=font_scale,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    cv_image = cv2.cvtColor(src=cv_image, code=cv2.COLOR_BGR2RGB)

    return cv_image


def initialize_persons(current_persons: List[dict], known_persons: List[dict]) -> None:
    for index, person in enumerate(known_persons):
        if IGNORE_STAFF and person["type"] == "STAFF":
            continue

        current_persons.append(
            {
                "id": index,
                "type": person["type"],
                "facial_area": None,
                "landmarks": {
                    "right_eye": None,
                    "left_eye": None,
                    "nose": None,
                    "mouth_right": None,
                    "mouth_left": None,
                },
                "face_data_usage_count": 0,
            }
        )


def calculate_facial_area(landmarks: dict) -> List[int]:
    # Initialize min and max values to None
    x_min, y_min, x_max, y_max = None, None, None, None

    # Iterate through the landmarks and update min and max values
    for landmark in landmarks.values():
        x, y = landmark
        if x_min is None or x < x_min:
            x_min = x
        if y_min is None or y < y_min:
            y_min = y
        if x_max is None or x > x_max:
            x_max = x
        if y_max is None or y > y_max:
            y_max = y

    # Apply the margin to the calculated bounding box
    x_min -= FACIAL_AREA_MARGIN
    y_min -= FACIAL_AREA_MARGIN
    x_max += FACIAL_AREA_MARGIN
    y_max += FACIAL_AREA_MARGIN

    return [x_min, y_min, x_max, y_max]


def update_persons(current_persons, headcount_change=None, detected_faces=None):
    # If the face data of entering or leaving persons has to be updated
    if headcount_change is not None:
        # Identifier of the person involved in the headcount change
        person_id = headcount_change["person"]["id"]
        # If the person entered the scene
        if headcount_change["type"] == "enters":
            # Person's facial landmarks that were manually obtained
            manual_landmarks = headcount_change["person"]["landmarks"]
            """
            Not sure if I should calculate a facial area, for now is not needed.
            It prevents the landmark distance calculation from being used.
            It would only be useful if the face isn't detected in the first frames.
            """
            # Calculate an approximation of the facial area and add it to the person's data
            # approx_facial_area = calculate_facial_area(manual_landmarks)
            # current_persons[person_id]["facial_area"] = approx_facial_area

            # Add the manually obtained landmarks to the person's data
            for landmark, coordinates in manual_landmarks.items():
                current_persons[person_id]["landmarks"][landmark] = coordinates
        # If the person left the scene
        elif headcount_change["type"] == "leaves":
            # Clear their facial_area
            current_persons[person_id]["facial_area"] = None
            # Clear their face landmarks
            old_landmarks = current_persons[person_id]["landmarks"]
            for landmark, coordinates in old_landmarks.items():
                current_persons[person_id]["landmarks"][landmark] = None
            current_persons[person_id]["face_data_usage_count"] = 0
    # If the face data of detected persons has to be updated
    elif detected_faces is not None:
        for face_id, detected_face in detected_faces.items():
            # Identifier of the person whose face was detected
            person_id = detected_face["person_id"]
            # Update the facial area and landmarks for the detected person
            current_persons[person_id]["facial_area"] = detected_face["facial_area"]
            for landmark, coordinates in detected_face["landmarks"].items():
                current_persons[person_id]["landmarks"][landmark] = coordinates
            current_persons[person_id]["face_data_usage_count"] = 1


def main():
    input_bag_path = Path(get_file_path(BAG_FILE_TYPE))

    start_time = time.time()

    # Get the directory part of the path
    input_directory_path = input_bag_path.parent

    # Add the new folder 'edits'
    output_directory_path = input_directory_path / "edits"

    if not output_directory_path.exists():
        output_directory_path.mkdir(parents=True)

    # Get the current timestamp and format it
    current_timestamp = datetime.now().strftime(format="%Y%m%dT%H%M%S")

    # Create the new filename with '_EDITED' appended before the extension
    output_bag_name = (
        input_bag_path.stem + "_EDIT_" + current_timestamp + input_bag_path.suffix
    )

    # Combine the new directory path and the new filename
    output_bag_path = output_directory_path / output_bag_name

    input_bag = rosbag.Bag(input_bag_path, "r")
    output_bag = rosbag.Bag(output_bag_path, "w")

    # Get the frame info of the color stream
    first_frame_number, last_frame_number = None, None
    color_msg_count = 0
    for _, msg, _ in input_bag.read_messages(topics=[COLOR_DATA_TOPIC]):
        # Increase color message count
        color_msg_count += 1

        # If it's the first message, save its frame number
        if color_msg_count == 1:
            first_frame_number = msg.header.seq

        # Remember the last frame number
        last_frame_number = msg.header.seq

    # FACES
    current_persons = []  # Updated data of the persons showing up in the color stream

    headcount_changes = []  # List of the headcount changes during the whole stream
    stream_segments = []  # Color stream segmented by headcount changes

    if recognizable_persons_col is not None:
        recognizable_persons = json.loads(recognizable_persons_col)

        # Initialize the list of persons
        initialize_persons(
            current_persons=current_persons,
            known_persons=recognizable_persons,
        )

        # Populate the headcount changes list
        for index, person in enumerate(recognizable_persons):
            # Skip considering the appearances of the hospital staff if requested
            if IGNORE_STAFF and person["type"] == "STAFF":
                continue

            # Add a headcount change for each endpoint of the frame interval
            for appearance in person["appearances"]:
                headcount_changes.append(
                    {
                        "frame": appearance["start"]["frame"],
                        "type": "enters",
                        "person": {
                            "id": index,
                            "landmarks": appearance["start"]["landmarks"],
                        },
                    }
                )
                headcount_changes.append(
                    {
                        "frame": appearance["end"]["frame"] + 1,
                        "type": "leaves",
                        "person": {
                            "id": index,
                        },
                    }
                )
        # Sort the headcount changes by frame number
        headcount_changes.sort(key=lambda endpoint: endpoint["frame"])

        # Define the segments of the color stream
        # Initialize variables
        current_frame_number = first_frame_number  # Current color stream frame number
        current_headcount = set()  # Indexes of the persons in the current segment

        # Process the headcount changes to segment the color stream
        for headcount_change in headcount_changes:
            if current_frame_number < headcount_change["frame"]:
                stream_segments.append(
                    {
                        "start_frame": current_frame_number,
                        "end_frame": headcount_change["frame"] - 1,
                        "headcount": sorted(current_headcount),
                    }
                )
                current_frame_number = headcount_change["frame"]
            if headcount_change["type"] == "enters":
                current_headcount.add(headcount_change["person"]["id"])
            elif headcount_change["type"] == "leaves":
                current_headcount.remove(headcount_change["person"]["id"])
        # Add the final segment if there are frames left
        if current_frame_number <= last_frame_number:
            stream_segments.append(
                {
                    "start_frame": current_frame_number,
                    "end_frame": last_frame_number,
                    "headcount": sorted(current_headcount),
                }
            )

    # Stream segmentation variables
    segment_index, current_stream_segment, bridge = None, None, None
    if recognizable_persons_col is not None:
        # Initialize the stream segment
        segment_index = 0
        current_stream_segment = stream_segments[segment_index]

        # Initialize the CvBridge
        bridge = CvBridge()

    # Color stream variables
    color_msg_counter = 0

    for topic, msg, t in input_bag.read_messages():

        if COLOR_DATA_TOPIC in topic:
            # Keep track of the current color message to print the progress of the process
            color_msg_counter += 1
            print(
                f"\rProcessed {color_msg_counter}/{color_msg_count} messages form the color stream",
                end="",
                flush=True,
            )

            if recognizable_persons_col is not None:
                # Update the current segment of the stream
                if msg.header.seq > current_stream_segment["end_frame"]:
                    segment_index += 1
                    current_stream_segment = stream_segments[segment_index]

                # Check if the current frame has a headcount change to update the persons
                for headcount_change in headcount_changes:
                    if msg.header.seq == headcount_change["frame"]:
                        # Insert or remove the facial data of the person involved in the headcount change
                        update_persons(
                            current_persons=current_persons,
                            headcount_change=headcount_change,
                        )

                # Only process the current message if persons appear in the stream segment
                if len(current_stream_segment["headcount"]) != 0:

                    # Convert ROS Image message to OpenCV image
                    cv_image = bridge.imgmsg_to_cv2(
                        img_msg=msg, desired_encoding="bgr8"
                    )

                    # Process the image using OpenCV
                    processed_cv_image = process_cv_image(
                        cv_image=cv_image,
                        current_segment=current_stream_segment,
                        current_persons=current_persons,
                        frame_number=msg.header.seq,
                    )

                    # Convert OpenCV image back to ROS Image message
                    msg = bridge.cv2_to_imgmsg(cvim=processed_cv_image, encoding="rgb8")

        output_bag.write(topic, msg, t)

    input_bag.close()
    output_bag.close()

    print(f"\nProcessing complete.\nEdited bag file saved to: {output_bag_path}")

    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f"Time: {format(round(duration,3))} minutes")


if __name__ == "__main__":
    main()
