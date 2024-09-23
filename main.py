import io
import json
import time
import tkinter
import yaml
from datetime import datetime
from pathlib import Path
from tkinter import filedialog
from typing import Tuple, List, Dict, Optional
import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
from genpy import Time
from retinaface import RetinaFace
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sys
import logging

# Suppress warnings from rospy.logwarn before importing rosbag
logging.getLogger("rosout").setLevel(logging.ERROR)
import rosbag

BAG_INITIALIZATION_TIME = Time(nsecs=1)  # Publish time of RealSense metadata
BAG_SENSORS_START_TIME = Time(nsecs=20000)  # Publish start time of sensors


def get_undesired_intervals(bag_metadata: pd.DataFrame, config: dict, min_output_bag_frames: float) -> List[dict]:
    undesired_intervals = []

    wound_partially_out = bag_metadata["wound_partially_out"].values[0]
    wound_fully_out = bag_metadata["wound_fully_out"].values[0]
    wound_blurry = bag_metadata["wound_blurry"].values[0]
    wound_covered = bag_metadata["wound_covered"].values[0]
    wound_near_face = bag_metadata["wound_near_face"].values[0]

    # Add intervals where wound visibility is partial based on tolerance
    if not pd.isna(wound_partially_out):
        partial_intervals = json.loads(wound_partially_out)

        if not config["KEEP_WOUND_PARTIALLY_OUT"]:
            filtered_intervals = partial_intervals
        else:
            filtered_intervals = [
                interval for interval in partial_intervals if interval["degree"] > config["PARTIALLY_OUT_TOLERANCE"]
            ]

        # Remove the 'degree' key from partial intervals
        for interval in filtered_intervals:
            interval.pop("degree", None)

        undesired_intervals += filtered_intervals

    # Add intervals where wound visibility is none
    if (not config["KEEP_WOUND_FULLY_OUT"]) and (not pd.isna(wound_fully_out)):
        undesired_intervals += json.loads(wound_fully_out)

    if (not config["KEEP_WOUND_BLURRY"]) and (not pd.isna(wound_blurry)):
        undesired_intervals += json.loads(wound_blurry)

    if (not config["KEEP_WOUND_COVERED"]) and (not pd.isna(wound_covered)):
        undesired_intervals += json.loads(wound_covered)

    if (not config["KEEP_WOUND_NEAR_FACE"]) and (not pd.isna(wound_near_face)):
        undesired_intervals += json.loads(wound_near_face)

    if not undesired_intervals:
        return []

    # Combine overlapping or adjacent intervals
    # . Sort intervals by start_frame
    undesired_intervals.sort(key=lambda x: x["start_frame"])

    combined_intervals = [undesired_intervals[0]]

    for current_interval in undesired_intervals[1:]:
        last_interval = combined_intervals[-1]

        # Calculate the number of desired frames between the current and the last undesired interval
        frames_in_gap = current_interval["start_frame"] - last_interval["end_frame"] - 1

        # If the current interval overlaps or is adjacent to the last one, merge them
        if (current_interval["start_frame"] <= last_interval["end_frame"] + 1) or (
            frames_in_gap < min_output_bag_frames
        ):
            last_interval["end_frame"] = max(last_interval["end_frame"], current_interval["end_frame"])
        else:
            combined_intervals.append(current_interval)

    return combined_intervals


def get_desired_intervals(
    undesired_intervals: List[dict],
    first_frame_number: int,
    last_frame_number: int,
    min_output_bag_frames: float,
) -> List[dict]:
    desired_intervals = []

    # Initialize the start frame for the first desired interval
    current_start = first_frame_number

    for undesired in undesired_intervals:
        # If there is a gap between the current start and the undesired interval, it's a desired interval
        if undesired["start_frame"] > current_start:
            desired_intervals.append(
                {
                    "start_frame": current_start,
                    "end_frame": undesired["start_frame"] - 1,
                }
            )
        # Move the start frame to the end of the current undesired interval + 1
        current_start = undesired["end_frame"] + 1

    # If there is a desired interval after the last undesired interval
    if current_start <= last_frame_number:
        desired_intervals.append(
            {
                "start_frame": current_start,
                "end_frame": last_frame_number,
            }
        )

    # Filter out desired intervals that are shorter than the minimum duration
    return [
        interval
        for interval in desired_intervals
        if (interval["end_frame"] - interval["start_frame"] + 1) >= min_output_bag_frames
    ]


def is_frame_in_interval(frame: int, interval: dict) -> bool:
    return interval["start_frame"] <= frame <= interval["end_frame"]


def is_tracking_required_for_interval(interval: dict, headcount_segments: List[dict]) -> bool:
    # Check headcount at start_frame
    for segment in headcount_segments:
        if segment["start_frame"] <= interval["start_frame"] <= segment["end_frame"]:
            if len(segment["headcount"]) > 0:
                return True
    # Check headcount at end_frame
    for segment in headcount_segments:
        if segment["start_frame"] <= interval["end_frame"] <= segment["end_frame"]:
            if len(segment["headcount"]) > 0:
                return True

    return False


def are_there_persons_in_interval(interval: dict, headcount_segments: List[dict]) -> bool:

    for segment in headcount_segments:
        # Check if the interval overlaps with this segment
        if segment["start_frame"] <= interval["end_frame"] and segment["end_frame"] >= interval["start_frame"]:
            # If headcount in the segment is greater than 0, there are people in the interval
            if len(segment["headcount"]) > 0:
                return True

    return False


def calculate_iou(box1: List[int], box2: List[int]) -> float:
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

    # Return IoU score
    return inter_area / union_area if union_area > 0 else 0


def calculate_avg_landmark_distance(
    person_landmarks: Dict[str, List[float]],
    detected_landmarks: Dict[str, List[float]],
) -> Optional[float]:
    total_distance = 0
    valid_landmark_count = 0

    for landmark, landmark_value in person_landmarks.items():
        if landmark_value is not None:
            detected_point = np.array(detected_landmarks[landmark])
            known_point = np.array(landmark_value)
            distance = np.linalg.norm(detected_point - known_point)
            total_distance += distance
            valid_landmark_count += 1

    return total_distance / valid_landmark_count if valid_landmark_count > 0 else None


def flip_face_coordinates(boundary_height: int, boundary_width: int, detected_face: dict, flip_code: int) -> dict:

    facial_area = detected_face["facial_area"]

    right_eye, left_eye, nose, mouth_right, mouth_left = None, None, None, None, None

    # Flip vertically
    if (flip_code == 0) or (flip_code == -1):
        # Invert the Y coordinates of facial_area
        facial_area_y2 = boundary_height - facial_area[1]
        facial_area_y1 = boundary_height - facial_area[3]
        facial_area[1] = facial_area_y1
        facial_area[3] = facial_area_y2

        # Invert the Y coordinates of landmarks
        right_eye = detected_face["landmarks"]["left_eye"]
        right_eye[1] = boundary_height - right_eye[1]
        left_eye = detected_face["landmarks"]["right_eye"]
        left_eye[1] = boundary_height - left_eye[1]
        nose = detected_face["landmarks"]["nose"]
        nose[1] = boundary_height - nose[1]
        mouth_right = detected_face["landmarks"]["mouth_left"]
        mouth_right[1] = boundary_height - mouth_right[1]
        mouth_left = detected_face["landmarks"]["mouth_right"]
        mouth_left[1] = boundary_height - mouth_left[1]

    # Flip horizontally
    if (flip_code == 1) or (flip_code == -1):
        # Invert the X coordinates of facial_area
        facial_area_x2 = boundary_width - facial_area[0]
        facial_area_x1 = boundary_width - facial_area[2]
        facial_area[0] = facial_area_x1
        facial_area[2] = facial_area_x2

        # Invert the X coordinates of landmarks
        right_eye = detected_face["landmarks"]["left_eye"]
        right_eye[0] = boundary_width - right_eye[0]
        left_eye = detected_face["landmarks"]["right_eye"]
        left_eye[0] = boundary_width - left_eye[0]
        nose = detected_face["landmarks"]["nose"]
        nose[0] = boundary_width - nose[0]
        mouth_right = detected_face["landmarks"]["mouth_left"]
        mouth_right[0] = boundary_width - mouth_right[0]
        mouth_left = detected_face["landmarks"]["mouth_right"]
        mouth_left[0] = boundary_width - mouth_left[0]

    return {
        "facial_area": facial_area,
        "landmarks": {
            "right_eye": right_eye,
            "left_eye": left_eye,
            "nose": nose,
            "mouth_right": mouth_right,
            "mouth_left": mouth_left,
        },
    }


def analyze_and_anonymize_frame(
    frame: np.ndarray,
    headcount_segment: dict,
    tracked_persons: List[dict],
    config: dict,
    should_anonymize: bool,
) -> Optional[np.ndarray]:

    frame_height, frame_width = frame.shape[:2]
    boundary_height = frame_height - 1
    boundary_width = frame_width - 1

    assigned_person_ids = set()

    # Detect faces using RetinaFace
    detected_faces = RetinaFace.detect_faces(img_path=frame, threshold=config["FACE_DETECTION_THRESHOLD"])

    if config["CHECK_ALL_FLIPS"] or (len(detected_faces) < len(headcount_segment["headcount"])):
        # Start from a horizontal flip
        current_flip_code = 1

        # Loop through all the flip codes
        while current_flip_code >= -1:

            # Flip the image and check again for detection
            flipped_frame = cv2.flip(src=frame, flipCode=current_flip_code)
            flipped_detected_faces = RetinaFace.detect_faces(
                img_path=flipped_frame,
                threshold=config["FACE_DETECTION_THRESHOLD"],
            )

            # If faces were detected on the flipped image
            if len(flipped_detected_faces) > 0:

                # Unflip all the detected faces and ad them if not repeated
                for flipped_detected_face in flipped_detected_faces.values():

                    unflipped_face = flip_face_coordinates(
                        boundary_height=boundary_height,
                        boundary_width=boundary_width,
                        detected_face=flipped_detected_face,
                        flip_code=current_flip_code,
                    )

                    is_new_face = True
                    for detected_face in detected_faces.values():
                        facial_area_iou = calculate_iou(
                            box1=unflipped_face["facial_area"],
                            box2=detected_face["facial_area"],
                        )

                        if facial_area_iou > config["MIN_FACIAL_AREA_IOU"]:
                            is_new_face = False

                    if is_new_face:
                        # Assemble the unflipped face and add it to the detected faces
                        detected_faces[f"face_{len(detected_faces) + 1}"] = {
                            "score": flipped_detected_face["score"],
                            "facial_area": unflipped_face["facial_area"],
                            "landmarks": {
                                "right_eye": unflipped_face["landmarks"]["right_eye"],
                                "left_eye": unflipped_face["landmarks"]["left_eye"],
                                "nose": unflipped_face["landmarks"]["nose"],
                                "mouth_right": unflipped_face["landmarks"]["mouth_right"],
                                "mouth_left": unflipped_face["landmarks"]["mouth_left"],
                            },
                        }

                if (not config["CHECK_ALL_FLIPS"]) and (len(detected_faces) >= len(headcount_segment["headcount"])):
                    # Exit while loop
                    break

            # Set the next flipCode
            current_flip_code -= 1

    # If faces were detected
    if len(detected_faces) > 0:

        # Iterate over a snapshot of the detected_faces's items
        for face_id, detected_face in list(detected_faces.items()):
            matched_person_id = None
            matched_person_type = None
            max_facial_area_iou = 0.0
            min_landmark_distance = float("inf")

            for person in tracked_persons:

                # Skip if this person's id is already assigned
                if person["id"] in assigned_person_ids:
                    continue

                # If this person's face data was manually defined and hasn't been used yet
                if person["face_manually_defined"] and (person["face_reuse_count"] == -1):

                    avg_landmark_distance = calculate_avg_landmark_distance(
                        person_landmarks=person["landmarks"],
                        detected_landmarks=detected_face["landmarks"],
                    )

                    if avg_landmark_distance is not None:
                        if avg_landmark_distance < config["MAX_LANDMARK_DISTANCE"]:
                            matched_person_id = person["id"]
                            matched_person_type = person["type"]
                            break

                elif person["facial_area"] is not None:
                    # Calculate IOU between the person's facial area and the detected facial area
                    facial_area_iou = calculate_iou(box1=person["facial_area"], box2=detected_face["facial_area"])

                    if facial_area_iou > max_facial_area_iou:
                        max_facial_area_iou = facial_area_iou
                        matched_person_id = person["id"]
                        matched_person_type = person["type"]

                        # Secondary check using landmarks to refine the match
                        if matched_person_id is not None:

                            avg_landmark_distance = calculate_avg_landmark_distance(
                                person_landmarks=person["landmarks"],
                                detected_landmarks=detected_face["landmarks"],
                            )

                            if avg_landmark_distance is not None:
                                if (
                                    avg_landmark_distance < config["MAX_LANDMARK_DISTANCE"]
                                    and avg_landmark_distance < min_landmark_distance
                                ):
                                    min_landmark_distance = avg_landmark_distance
                                else:
                                    matched_person_id = None
                                    matched_person_type = None

            if matched_person_id is not None:

                # Save the person's tracking info in the face dict
                detected_face["area_origin"] = "AUTO"
                detected_face["area_reuse_count"] = 0
                detected_face["person_id"] = matched_person_id
                detected_face["person_type"] = matched_person_type

                # Add the matched person's id to the assigned set
                assigned_person_ids.add(matched_person_id)
            else:
                # Remove the detected face if no good person match is found
                del detected_faces[face_id]

        # If not all detected faces were deleted
        if len(detected_faces) > 0:
            # Update the face data of the persons present in the segment
            update_tracked_persons(
                tracked_persons=tracked_persons,
                detected_faces=detected_faces,
            )

    if should_anonymize:

        # Check for missing persons
        missing_person_ids = set(headcount_segment["headcount"]) - assigned_person_ids

        # If there are persons missing in the detection list
        if len(missing_person_ids) > 0:

            # Iterate through the missing persons to check if their previous facial data exists and can be used
            for index, person_id in enumerate(missing_person_ids, start=1):

                # If the reuse limit for the same face data hasn't been reached yet
                if (not config["LIMIT_FACE_REUSE"]) or (
                    tracked_persons[person_id]["face_reuse_count"] < config["MAX_FACE_REUSE_COUNT"]
                ):
                    tracked_persons[person_id]["face_reuse_count"] += 1

                    detected_faces[f"previous_{index}"] = {
                        "score": 1.0,
                        "facial_area": tracked_persons[person_id]["facial_area"],
                        "landmarks": tracked_persons[person_id]["landmarks"],
                        "area_origin": ("MANUAL" if tracked_persons[person_id]["face_manually_defined"] else "AUTO"),
                        "area_reuse_count": tracked_persons[person_id]["face_reuse_count"],
                        "person_id": person_id,
                        "person_type": tracked_persons[person_id]["type"],
                    }

        if len(detected_faces) > 0:
            # Blur each detected face
            for detected_face in detected_faces.values():
                x1, y1, x2, y2 = detected_face["facial_area"]

                # Add padding to the anonymization box making sure it doesn't extend beyond the frame boundary
                x1_padded = max(x1 - config["BOX_PADDING"], 0)
                y1_padded = max(y1 - config["BOX_PADDING"], 0)
                x2_padded = min(x2 + config["BOX_PADDING"], boundary_width)
                y2_padded = min(y2 + config["BOX_PADDING"], boundary_height)

                # Remove gap between facial_area and frame boundary to make sure no face parts are visible
                if x1_padded < config["BOUNDARY_PROXIMITY_LIMIT"]:
                    x1_padded = 0
                if y1_padded < config["BOUNDARY_PROXIMITY_LIMIT"]:
                    y1_padded = 0
                if x2_padded > boundary_width - config["BOUNDARY_PROXIMITY_LIMIT"]:
                    x2_padded = boundary_width
                if y2_padded > boundary_height - config["BOUNDARY_PROXIMITY_LIMIT"]:
                    y2_padded = boundary_height

                if config["BLUR_BOX"]:
                    frame[y1_padded : (y2_padded + 1), x1_padded : (x2_padded + 1)] = cv2.GaussianBlur(
                        src=frame[y1_padded : (y2_padded + 1), x1_padded : (x2_padded + 1)],
                        ksize=tuple(config["BLUR_KERNEL_SIZE"]),
                        sigmaX=config["BLUR_SIGMA"],
                    )
                else:
                    frame[y1_padded : (y2_padded + 1), x1_padded : (x2_padded + 1)] = config["BOX_COLOR"]

                if config["DISPLAY_TRACKING_INFO"]:

                    person_id = detected_face["person_id"]
                    person_type = detected_face["person_type"]
                    area_origin = detected_face["area_origin"]
                    area_reuse_count = detected_face["area_reuse_count"]

                    # Format text in the desired multi-line structure
                    info_lines = [
                        f"ID: {person_id}",
                        # f"Type: {person_type.lower()}",
                        f"Origin: {area_origin.lower()}",
                        f"Reuses: {area_reuse_count}",
                    ]

                    font_face = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = config["TRACKING_INFO_FONT_SIZE"]
                    line_thickness = config["TRACKING_INFO_FONT_WEIGHT"]

                    text_y = y1_padded + 18
                    text_x = x1_padded + 9

                    # Render each line of text with proper spacing
                    for line in info_lines:
                        text_size, _ = cv2.getTextSize(
                            text=line,
                            fontFace=font_face,
                            fontScale=font_scale,
                            thickness=line_thickness,
                        )

                        # Draw each info line
                        cv2.putText(
                            img=frame,
                            text=line,
                            org=(text_x, text_y),
                            fontFace=font_face,
                            fontScale=font_scale,
                            color=tuple(config["TRACKING_INFO_FONT_COLOR"]),
                            thickness=line_thickness,
                            lineType=cv2.LINE_AA,
                        )

                        # Update Y position for the next line
                        text_y += text_size[1] + 6  # Adding 5 pixels of vertical spacing between lines

        # Revert the color space back to RGB8
        frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

        return frame

    else:
        return None


def initialize_tracked_persons(
    tracked_persons: List[dict],
    person_appearances: List[dict],
    ignore_staff: bool,
) -> None:
    for index, person in enumerate(person_appearances):
        if ignore_staff and person["type"] == "STAFF":
            continue

        tracked_persons.append(
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
                "face_manually_defined": False,
                "face_reuse_count": 0,
            }
        )


def update_tracked_persons(
    tracked_persons: List[dict],
    headcount_change: dict = None,
    detected_faces: dict = None,
    config: dict = None,
) -> None:
    # If the face data of entering or leaving persons has to be updated
    if headcount_change is not None:
        # Identifier of the person involved in the headcount change
        person_id = headcount_change["person"]["id"]

        # If the person entered the scene
        if headcount_change["type"] == "enters":
            # Person's facial landmarks that were manually obtained
            manual_landmarks = headcount_change["person"]["landmarks"]
            # Create an approximation of the facial area and add it to the person's data
            tracked_persons[person_id]["facial_area"] = approximate_facial_area(
                landmarks=manual_landmarks,
                config=config,
            )
            # Add the manually obtained landmarks to the person's data
            for landmark, coordinates in manual_landmarks.items():
                tracked_persons[person_id]["landmarks"][landmark] = coordinates
            # Mark the face data as user defined
            tracked_persons[person_id]["face_manually_defined"] = True
            # Initialize the number of uses
            tracked_persons[person_id]["face_reuse_count"] = -1

        # If the person left the scene
        elif headcount_change["type"] == "leaves":
            # Reset the facial_area
            tracked_persons[person_id]["facial_area"] = None
            # Reset the face landmarks
            old_landmarks = tracked_persons[person_id]["landmarks"]
            for landmark, coordinates in old_landmarks.items():
                tracked_persons[person_id]["landmarks"][landmark] = None
            # Reset the face origin
            tracked_persons[person_id]["face_manually_defined"] = False
            # Reset the face reuse count
            tracked_persons[person_id]["face_reuse_count"] = 0

    # If the face data of detected persons has to be updated
    elif detected_faces is not None:
        for face_id, detected_face in detected_faces.items():
            # Identifier of the person whose face was detected
            person_id = detected_face["person_id"]
            # Update the facial area and landmarks for the detected person
            tracked_persons[person_id]["facial_area"] = detected_face["facial_area"]
            for landmark, coordinates in detected_face["landmarks"].items():
                tracked_persons[person_id]["landmarks"][landmark] = coordinates
            # Mark the face data as detected by the model
            tracked_persons[person_id]["face_manually_defined"] = False
            # Reset the reuse count
            tracked_persons[person_id]["face_reuse_count"] = 0


def approximate_facial_area(landmarks: dict, config: dict) -> List[int]:
    # Initialize min and max values to None
    x_min, y_min, x_max, y_max = None, None, None, None

    # Iterate through the landmarks and update min and max values
    for landmark in landmarks.values():
        x, y = landmark
        if (x_min is None) or (x < x_min):
            x_min = x
        if (y_min is None) or (y < y_min):
            y_min = y
        if (x_max is None) or (x > x_max):
            x_max = x
        if (y_max is None) or (y > y_max):
            y_max = y

    # Apply the margin to the calculated bounding box
    x_min -= config["MANUAL_AREA_PADDING"]
    y_min -= config["MANUAL_AREA_PADDING"]
    x_max += config["MANUAL_AREA_PADDING"]
    y_max += config["MANUAL_AREA_PADDING"]

    return [x_min, y_min, x_max, y_max]


def load_config_file(filename: str) -> dict:
    try:
        # Load configuration from the YAML file
        with open(filename, "r") as config_file:
            config = yaml.safe_load(config_file)

        # Check if the config is None or empty
        if config is None or not config:
            print(f"\nError: The config file '{filename}' was loaded, but it contains no data.")
        else:
            return config

    except FileNotFoundError:
        print(f"\nError: The config file '{filename}' was not found.")
    except yaml.YAMLError as e:
        print(f"\nError: An issue occurred while parsing the config file '{filename}'.\n\tDetails: {e}")
    except Exception as e:
        print(f"\nError: An issue occurred while loading the config file '{filename}'.\n\tDetails: {e}")

    # Exit the program
    print(f"\nExiting the program...")
    sys.exit()


def load_metadata_file(filename: str) -> DataFrame:
    try:
        # Load the Excel file containing the dataset metadata
        with pd.ExcelFile("dataset_metadata.xlsx") as xlsx:
            dataset_metadata = pd.read_excel(xlsx)

        # Check if the DataFrame is empty
        if dataset_metadata.empty:
            print(f"\nError: The metadata file '{filename}' was loaded, but it contains no data.")
        else:
            return dataset_metadata

    except FileNotFoundError:
        print(f"\nError: The metadata file '{filename}' was not found.")
    except Exception as e:
        print(f"\nError: An issue occurred while loading the metadata file '{filename}'.\n\tDetails: {e}")

    # Exit the program
    print(f"\nExiting the program...")
    sys.exit()


def select_dataset_path(project_path: Path, window: tkinter.Tk) -> Optional[Path]:

    dataset_path = filedialog.askdirectory(
        initialdir=project_path,
        mustexist=True,
        parent=window,
        title="Select Dataset Folder",
    )

    if dataset_path == "":
        return None
    else:
        return Path(dataset_path)


def create_output_bag(
    input_bag_path: Path,
    run_timestamp: str,
    output_bag_clip: int,
    output_directory_path: Path,
) -> Tuple[rosbag.Bag, Path]:

    output_bag_name = (
        input_bag_path.stem + "_EDIT-" + run_timestamp + "_CLIP-" + str(output_bag_clip) + input_bag_path.suffix
    )

    # Combine the new directory path and the new filename
    output_bag_path = output_directory_path / output_bag_name

    return rosbag.Bag(f=output_bag_path, mode="w"), output_bag_path


def process_bag(config: dict, input_bag_path: Path, input_bag_metadata: DataFrame, output_directory_path: Path):

    bag_processing_start_time = time.time()

    print(f"\nOpening bag file...")
    input_bag = rosbag.Bag(f=input_bag_path, mode="r")
    print(f"Bag file opened successfully.")

    print(f"\nRetrieving camera info...")
    camera_name = None
    camera_firmware = None
    for _, msg, _ in input_bag.read_messages(topics=["/device_0/info"], end_time=BAG_INITIALIZATION_TIME):
        if msg.key == "Name":
            camera_name = msg.value
        elif msg.key == "Firmware Version":
            camera_firmware = msg.value
        if camera_name and camera_firmware:
            break
    print(f"Camera info:")
    print(f"\t- Name: {camera_name}")
    print(f"\t- Firmware: {camera_firmware}")

    print("\nRetrieving color stream info...")
    # Retrieve the color stream's data and info topics
    color_data_topic = None
    color_info_topic = None
    for topic in input_bag.get_type_and_topic_info().topics.keys():
        if topic.endswith("Color_0/info"):
            color_info_topic = topic
        elif "Color_0/image/data" in topic:
            color_data_topic = topic
        if color_data_topic and color_info_topic:
            break

    # Retrieve the color stream's resolution, encoding, FPS, and the first and last frame numbers with their timestamps
    color_stream_resolution = None
    color_stream_encoding = None
    color_stream_fps = None
    first_frame_number = None
    last_frame_number = None
    first_frame_timestamp = None
    last_frame_timestamp = None
    for topic, msg, _ in input_bag.read_messages(topics=[color_data_topic, color_info_topic]):
        if topic == color_info_topic:
            color_stream_encoding = msg.encoding
            color_stream_fps = msg.fps
        elif color_data_topic in topic:
            if first_frame_number is None:
                first_frame_number = msg.header.seq
                first_frame_timestamp = msg.header.stamp
                color_stream_resolution = f"{msg.width}x{msg.height}"
            last_frame_number = msg.header.seq
            last_frame_timestamp = msg.header.stamp

    # Calculate the color stream's duration using the first and last timestamps
    color_stream_duration = None
    if first_frame_timestamp and last_frame_timestamp:
        color_stream_duration = (last_frame_timestamp - first_frame_timestamp).to_sec()

    print(f"Color stream info:")
    # print(f"\t- Data topic: {color_data_topic}")
    # print(f"\t- Info topic: {color_info_topic}")
    print(f"\t- Resolution: {color_stream_resolution}")
    print(f"\t- Encoding: {color_stream_encoding.upper()}")
    print(f"\t- FPS: {color_stream_fps}")
    print(f"\t- Duration: {round(color_stream_duration, 2)} seconds")
    print(f"\t- Frames:")
    print(f"\t\t· First: {first_frame_number}")
    print(f"\t\t· Last: {last_frame_number}")

    # Minimum number of frames in an output bag
    min_output_bag_frames = color_stream_fps * config["MIN_OUTPUT_DURATION"]

    print("\nCalculating undesired color frame intervals...")
    # Get the intervals to discard
    undesired_intervals = get_undesired_intervals(
        bag_metadata=input_bag_metadata,
        config=config,
        min_output_bag_frames=min_output_bag_frames,
    )
    if len(undesired_intervals) > 0:
        print("Undesired color frame intervals:")
        for interval in undesired_intervals:
            print(f"\t- {interval}")
    else:
        print("No undesired color frame intervals were found")

    print("\nCalculating desired color frame intervals...")
    # Get the intervals to save
    desired_intervals = get_desired_intervals(
        undesired_intervals=undesired_intervals,
        first_frame_number=first_frame_number,
        last_frame_number=last_frame_number,
        min_output_bag_frames=min_output_bag_frames,
    )

    if len(desired_intervals) > 0:
        print("Desired color frame intervals:")
        for interval in desired_intervals:
            print(f"\t- {interval}")

        # Get the current timestamp and format it
        run_timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

        # Initialize data structures for color stream processing
        tracked_persons = []  # Updated data of the persons appearing
        headcount_changes = []  # List of the headcount changes during the stream
        headcount_segments = []  # Stream segmented by headcount changes
        # Process person appearances if available
        person_appearances_cell = input_bag_metadata["person_appearances"].values[0]
        person_appearances = None
        if not pd.isna(person_appearances_cell):
            person_appearances = json.loads(person_appearances_cell)

        if person_appearances is not None:

            # Initialize the list of tracked persons
            initialize_tracked_persons(
                tracked_persons=tracked_persons,
                person_appearances=person_appearances,
                ignore_staff=config["IGNORE_STAFF"],
            )

            # Populate the headcount changes list
            for index, person in enumerate(person_appearances):
                # Skip considering the appearances of the hospital staff if requested
                if config["IGNORE_STAFF"] and person["type"] == "STAFF":
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
            current_frame_number = first_frame_number
            current_headcount = set()  # Indexes of the persons in the current segment

            # Process the headcount changes to segment the color stream
            for headcount_change in headcount_changes:
                if current_frame_number < headcount_change["frame"]:
                    headcount_segments.append(
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
                headcount_segments.append(
                    {
                        "start_frame": current_frame_number,
                        "end_frame": last_frame_number,
                        "headcount": sorted(current_headcount),
                    }
                )

        # Stream segmentation by headcount
        headcount_segment_index = None
        current_headcount_segment = None
        bridge = None
        if person_appearances is not None:
            # Initialize the stream segment
            headcount_segment_index = 0
            current_headcount_segment = headcount_segments[headcount_segment_index]

            # Initialize the CvBridge for ROS-OpenCV conversions
            bridge = CvBridge()

        desired_interval_index = 0
        current_desired_interval = desired_intervals[desired_interval_index]
        desired_interval_has_persons = are_there_persons_in_interval(
            interval=current_desired_interval,
            headcount_segments=headcount_segments,
        )

        undesired_interval_index = None
        current_undesired_interval = None
        undesired_interval_has_persons = False
        undesired_interval_needs_tracking = False
        if len(undesired_intervals) > 0:
            undesired_interval_index = 0
            current_undesired_interval = undesired_intervals[undesired_interval_index]
            if person_appearances is not None:
                #
                undesired_interval_has_persons = are_there_persons_in_interval(
                    interval=current_undesired_interval,
                    headcount_segments=headcount_segments,
                )
                #
                undesired_interval_needs_tracking = is_tracking_required_for_interval(
                    interval=current_undesired_interval,
                    headcount_segments=headcount_segments,
                )

        writing_desired_frames = False
        clip_start_time = None
        output_bag = None
        frame_number = None
        frame_action = None
        processed_cv_image = None
        # Iterate through all messages in the input bag file
        for topic, msg_raw, t in input_bag.read_messages(raw=True):

            if color_data_topic in topic:

                # Unpack the message tuple
                msg_type, serialized_bytes, md5sum, pos, pytype = msg_raw

                # Deserialize the message bytes into an Image message
                deserialized_bytes = Image()
                deserialized_bytes.deserialize(serialized_bytes)

                # Save the header values from the message
                frame_number = deserialized_bytes.header.seq
                frame_timestamp = deserialized_bytes.header.stamp

                # Update the current desired interval
                if (frame_number > current_desired_interval["end_frame"]) and (
                    desired_interval_index < len(desired_intervals) - 1
                ):
                    desired_interval_index += 1
                    current_desired_interval = desired_intervals[desired_interval_index]
                    #
                    if person_appearances is not None:
                        desired_interval_has_persons = are_there_persons_in_interval(
                            interval=current_desired_interval,
                            headcount_segments=headcount_segments,
                        )

                # Update the current undesired interval
                elif (
                    (len(undesired_intervals) > 0)
                    and (frame_number > current_undesired_interval["end_frame"])
                    and (undesired_interval_index < len(undesired_intervals) - 1)
                ):
                    undesired_interval_index += 1
                    current_undesired_interval = undesired_intervals[undesired_interval_index]
                    #
                    if person_appearances is not None:
                        #
                        undesired_interval_has_persons = are_there_persons_in_interval(
                            interval=current_undesired_interval,
                            headcount_segments=headcount_segments,
                        )
                        # Update tracking requirement
                        undesired_interval_needs_tracking = is_tracking_required_for_interval(
                            interval=current_undesired_interval,
                            headcount_segments=headcount_segments,
                        )

                # If the current color frame is part of a desired interval
                frame_in_desired_interval = False
                if is_frame_in_interval(frame=frame_number, interval=current_desired_interval):
                    frame_in_desired_interval = True

                # Notify the start of a desired interval
                if frame_number == current_desired_interval["start_frame"]:

                    frame_action = "Saving"
                    interval_start = str(current_desired_interval["start_frame"])
                    interval_end = str(current_desired_interval["end_frame"])

                    interval_action = "..."
                    if (person_appearances is not None) and desired_interval_has_persons:
                        interval_action = "- Tracking and anonymizing persons in desired color frames..."

                    if frame_number != first_frame_number:
                        print()

                    print(f"\nProcessing interval [{interval_start}, {interval_end}] {interval_action}")

                # Notify the start of an undesired interval
                elif (len(undesired_intervals) > 0) and (frame_number == current_undesired_interval["start_frame"]):

                    frame_action = "Reading"
                    interval_start = str(current_undesired_interval["start_frame"])
                    interval_end = str(current_undesired_interval["end_frame"])

                    interval_action = "..."
                    if (
                        person_appearances is not None
                        and undesired_interval_has_persons
                        and undesired_interval_needs_tracking
                    ):
                        interval_action = "- Tracking persons in undesired color frames..."

                    print(f"\nProcessing interval [{interval_start}, {interval_end}] {interval_action}")

                if person_appearances is not None:
                    # Update the current headcount segment of the stream
                    if frame_number > current_headcount_segment["end_frame"]:
                        headcount_segment_index += 1
                        current_headcount_segment = headcount_segments[headcount_segment_index]

                    if frame_in_desired_interval or (
                        not frame_in_desired_interval and undesired_interval_needs_tracking
                    ):
                        # Check if the current frame has a headcount change to update the persons
                        for headcount_change in headcount_changes:
                            if frame_number == headcount_change["frame"]:
                                # Insert or remove the facial data of the person involved in the headcount change
                                update_tracked_persons(
                                    tracked_persons=tracked_persons,
                                    headcount_change=headcount_change,
                                    config=config,
                                )

                        # Process the image there are persons in it to track and/or anonymize
                        if len(current_headcount_segment["headcount"]) != 0:
                            # Convert the ROS Image message to an OpenCV image
                            cv_image = bridge.imgmsg_to_cv2(img_msg=deserialized_bytes, desired_encoding="bgr8")

                            # Process the image to keep track of the persons appearing and blur their faces
                            processed_cv_image = analyze_and_anonymize_frame(
                                frame=cv_image,
                                headcount_segment=current_headcount_segment,
                                tracked_persons=tracked_persons,
                                config=config,
                                should_anonymize=frame_in_desired_interval,
                            )

                if frame_in_desired_interval:

                    if person_appearances is not None:
                        # Only process the current message if persons appear in the stream segment
                        if len(current_headcount_segment["headcount"]) != 0:
                            # Convert the OpenCV image back to a ROS Image message
                            processed_ros_image = bridge.cv2_to_imgmsg(cvim=processed_cv_image, encoding="rgb8")

                            # Restore message header. Skip restoring frame_id to prevent issues with newer SDK versions
                            processed_ros_image.header.seq = frame_number
                            processed_ros_image.header.stamp = frame_timestamp

                            # Serialize the processed Image message
                            buffer = io.BytesIO()
                            processed_ros_image.serialize(buffer)

                            # Turn the original message tuple into a list to allow component editing
                            msg_raw_components = list(msg_raw)

                            # Replace the original serialized bytes with the processed ones
                            msg_raw_components[1] = buffer.getvalue()

                            # Turn the list back into a tuple
                            msg_raw = tuple(msg_raw_components)

                    # If a clip isn't being written
                    if not writing_desired_frames:
                        # Update the flag to enable the writing
                        writing_desired_frames = True

                        # Create a new output bag clip
                        output_bag, current_output_bag_path = create_output_bag(
                            input_bag_path=input_bag_path,
                            run_timestamp=run_timestamp,
                            output_bag_clip=desired_interval_index,
                            output_directory_path=output_directory_path,
                        )
                        print(f"\tCreated output bag file: {current_output_bag_path.name}")

                        # Write the RealSense config messages to the output bag clip
                        for rs_topic, rs_msg, rs_t in input_bag.read_messages(
                            end_time=BAG_INITIALIZATION_TIME,
                            raw=True,
                        ):
                            output_bag.write(topic=rs_topic, msg=rs_msg, t=rs_t, raw=True)

                print(f"\r\t{frame_action} color frame: {frame_number}", end="", flush=True)

            # If a clip is being written
            if writing_desired_frames:
                # save its start time if it wasn't yet found
                if clip_start_time is None:
                    clip_start_time = t

                # Write with the applied time offset
                output_bag.write(
                    topic=topic,
                    msg=msg_raw,
                    t=(t - clip_start_time + BAG_SENSORS_START_TIME),
                    raw=True,
                )

                if frame_number == current_desired_interval["end_frame"]:
                    # Close the current output bag clip
                    output_bag.close()
                    print(f"\n\tOutput bag file closed successfully.")

                    # If all the desired intervals were written the task is finished
                    if desired_interval_index == len(desired_intervals) - 1:
                        break

                    # Update the flag to disable the writing
                    writing_desired_frames = False

                    # Reset for the next clip
                    clip_start_time = None

    else:
        print("No desired color frame intervals were found")

    # Close the input bag file
    input_bag.close()
    print(f"\nFinished processing the file: {input_bag_path.name}")
    # Calculate and display the duration of the process
    bag_processing_duration = (time.time() - bag_processing_start_time) / 60
    print(f"\nElapsed time: {format(round(bag_processing_duration, 2))} minutes")


def main():
    # Initialize the Tkinter main window and hide it
    main_window = tkinter.Tk()
    main_window.withdraw()

    # Load configuration from the YAML file
    config = load_config_file("config.yaml")

    # Load the Excel file containing the dataset metadata
    dataset_metadata = load_metadata_file("dataset_metadata.xlsx")

    # Project path
    project_path = Path().resolve()

    # Open a file dialog to select the BAG file
    dataset_path = select_dataset_path(
        project_path=project_path,
        window=main_window,
    )

    # Exit the program if no file was selected
    if dataset_path is None:
        print("\nError: The dataset folder was not selected.")

        # Exit the program
        print(f"\nExiting the program...")
        sys.exit()
    else:
        print(f"\nSelected dataset folder: {dataset_path}")

    # Create the output folder if it doesn't exist
    output_directory_name = f"{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}"
    output_directory_path = dataset_path / config["OUTPUT_FOLDER_NAME"] / output_directory_name
    if not output_directory_path.exists():
        output_directory_path.mkdir(parents=True)
        print(
            f"\nCreated folder to save the output bag files: ./{config['OUTPUT_FOLDER_NAME']}/{output_directory_name}"
        )

    if dataset_path is not None:
        print("\n------------------------------------------------------------")

    # List all .bag files in the directory
    bag_file_paths = [bag_file_path for bag_file_path in dataset_path.glob("*.bag") if bag_file_path.is_file()]
    total_bag_files = len(bag_file_paths)

    if total_bag_files == 0:
        print("\nError: No bag files were found in the dataset folder.")

        # Exit the program
        print(f"\nExiting the program...")
        sys.exit()

    # Temporary benchmark for performance measurement
    dataset_processing_start_time = time.time()

    for file_index, bag_file_path in enumerate(bag_file_paths, start=1):
        # Extract the file name from the selected path
        bag_file_name = bag_file_path.name

        if total_bag_files > 1:
            print(f"\nStarting to process file {file_index}/{total_bag_files}: {bag_file_name}")
        else:
            print(f"\nStarting to process the file: {bag_file_name}")

        bag_metadata = dataset_metadata[dataset_metadata["bag_filename"] == bag_file_name]

        process_bag(
            config=config,
            input_bag_path=bag_file_path,
            input_bag_metadata=bag_metadata,
            output_directory_path=output_directory_path,
        )
        print("\n------------------------------------------------------------")

    if total_bag_files > 1:
        # Calculate and display the duration of the process
        dataset_processing_duration = (time.time() - dataset_processing_start_time) / 60
        print(f"\nTotal processing time: {round(dataset_processing_duration, 2)} minutes")


if __name__ == "__main__":
    main()
