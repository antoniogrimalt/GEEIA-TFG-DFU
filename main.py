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


def combine_undesired_intervals(
    intervals: List[dict],
    min_output_bag_frames: float,
) -> List[dict]:

    if not intervals:
        return []

    # Sort intervals by start_frame
    intervals.sort(key=lambda x: x["start_frame"])

    combined_intervals = [intervals[0]]

    for current_interval in intervals[1:]:
        last_interval = combined_intervals[-1]

        # Calculate the separation between the current and last intervals
        frame_gap = current_interval["start_frame"] - last_interval["end_frame"]

        # If the current interval overlaps or is adjacent to the last one, merge them
        if (current_interval["start_frame"] <= last_interval["end_frame"] + 1) or (
            frame_gap < min_output_bag_frames
        ):
            last_interval["end_frame"] = max(
                last_interval["end_frame"], current_interval["end_frame"]
            )
        else:
            combined_intervals.append(current_interval)

    return combined_intervals


def get_undesired_intervals(
    bag_metadata: pd.DataFrame,
    config: dict,
    min_output_bag_frames: float,
) -> List[dict]:

    wound_partially_out = bag_metadata["wound_partially_out"].values[0]
    wound_fully_out = bag_metadata["wound_fully_out"].values[0]
    wound_blurry = bag_metadata["wound_blurry"].values[0]
    wound_covered = bag_metadata["wound_covered"].values[0]
    wound_near_face = bag_metadata["wound_near_face"].values[0]

    undesired_intervals = []

    # Add intervals where wound visibility is partial based on tolerance
    if not pd.isna(wound_partially_out):
        partial_intervals = json.loads(wound_partially_out)

        if not config["KEEP_WOUND_PARTIALLY_OUT"]:
            filtered_intervals = partial_intervals
        else:
            filtered_intervals = [
                interval
                for interval in partial_intervals
                if interval["degree"] > config["PARTIALLY_OUT_TOLERANCE"]
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

    # Combine overlapping or adjacent undesired intervals
    return combine_undesired_intervals(
        intervals=undesired_intervals,
        min_output_bag_frames=min_output_bag_frames,
    )


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
    filtered_desired_intervals = [
        interval
        for interval in desired_intervals
        if (interval["end_frame"] - interval["start_frame"] + 1)
        >= min_output_bag_frames
    ]

    return filtered_desired_intervals


def is_frame_in_intervals(frame: int, intervals: List[dict]) -> bool:

    for interval in intervals:
        if interval["start_frame"] <= frame <= interval["end_frame"]:
            return True

    return False


def is_frame_in_interval(frame: int, interval: dict) -> bool:

    if interval["start_frame"] <= frame <= interval["end_frame"]:
        return True
    else:
        return False


def create_output_bag(
    input_bag_path: Path,
    run_timestamp: str,
    output_bag_clip: int,
    output_directory_path: Path,
) -> Tuple[rosbag.Bag, Path]:

    output_bag_name = (
        input_bag_path.stem
        + "_EDIT-"
        + run_timestamp
        + "_CLIP-"
        + str(output_bag_clip)
        + input_bag_path.suffix
    )

    # Combine the new directory path and the new filename
    output_bag_path = output_directory_path / output_bag_name

    return rosbag.Bag(output_bag_path, "w"), output_bag_path


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


def calculate_avg_landmark_distance(
    person_landmarks: Dict[str, List[float]],
    detected_landmarks: Dict[str, List[float]],
) -> Optional[float]:
    """
    Calculate the average distance between corresponding landmarks of a person and a detected face.

    Parameters:
    person_landmarks (Dict[str, List[float]]): A dictionary of known landmarks for a person.
                                               Keys are landmark names (e.g., 'right_eye', 'nose'), and values are [x, y] coordinates.
    detected_landmarks (Dict[str, List[float]]): A dictionary of detected landmarks from the current face.
                                                 Keys are landmark names (e.g., 'right_eye', 'nose'), and values are [x, y] coordinates.

    Returns:
    Optional[float]: The average distance between corresponding landmarks, or None if no valid landmarks are found.
    """
    total_distance = 0
    valid_landmark_count = 0

    for landmark, landmark_value in person_landmarks.items():
        if landmark_value is not None:
            detected_point = np.array(detected_landmarks[landmark])

            known_point = np.array(landmark_value)

            distance = np.linalg.norm(detected_point - known_point)

            total_distance += distance
            valid_landmark_count += 1

    if valid_landmark_count > 0:
        average_distance = total_distance / valid_landmark_count
        return average_distance
    else:
        return None


def analyze_and_anonymize_frame(
    frame: np.ndarray,
    headcount_segment: dict,
    tracked_persons: List[dict],
    config: dict,
    should_anonymize: bool,
) -> Optional[np.ndarray]:

    frame_height, frame_width = frame.shape[:2]

    assigned_person_ids = set()

    # Detect faces using RetinaFace
    detected_faces = RetinaFace.detect_faces(
        img_path=frame,
        threshold=config["FACE_DETECTION_THRESHOLD"],
    )

    # If no faces were detected
    if len(detected_faces) == 0:
        # Flip the image vertically and check again for detection
        flipped_frame = cv2.flip(src=frame, flipCode=0)
        flipped_detected_faces = RetinaFace.detect_faces(
            img_path=flipped_frame,
            threshold=config["FACE_DETECTION_THRESHOLD"],
        )
        # If faces were detected on the flipped image
        if len(flipped_detected_faces) > 0:

            # Unflip all the detected faces correcting its features
            for face_id, detected_face in flipped_detected_faces.items():
                # facial_area
                facial_area = detected_face["facial_area"]
                y2 = frame_height - facial_area[1]
                y1 = frame_height - facial_area[3]
                facial_area[1] = y1
                facial_area[3] = y2
                # right_eye
                right_eye = detected_face["landmarks"]["left_eye"]
                right_eye[1] = frame_height - right_eye[1]
                # left_eye
                left_eye = detected_face["landmarks"]["right_eye"]
                left_eye[1] = frame_height - left_eye[1]
                # nose
                nose = detected_face["landmarks"]["nose"]
                nose[1] = frame_height - nose[1]
                # mouth_right
                mouth_right = detected_face["landmarks"]["mouth_left"]
                mouth_right[1] = frame_height - mouth_right[1]
                # mouth_left
                mouth_left = detected_face["landmarks"]["mouth_right"]
                mouth_left[1] = frame_height - mouth_left[1]

                # Assemble the unflipped face and add it to the detected faces
                detected_faces[face_id] = {
                    "score": detected_face["score"],
                    "facial_area": facial_area,
                    "landmarks": {
                        "right_eye": right_eye,
                        "left_eye": left_eye,
                        "nose": nose,
                        "mouth_right": mouth_right,
                        "mouth_left": mouth_left,
                    },
                }

    # If faces were detected
    if len(detected_faces) > 0:

        # Iterate over a snapshot of the detected_faces's items
        for face_id, detected_face in list(detected_faces.items()):
            matched_person_id = None
            max_facial_area_iou = 0.0
            min_landmark_distance = float("inf")

            for person in tracked_persons:

                # Skip if this person's id is already assigned
                if person["id"] in assigned_person_ids:
                    continue

                # If this person's face data hasn't been updated yet
                if person["facial_area"] is None:

                    avg_landmark_distance = calculate_avg_landmark_distance(
                        person_landmarks=person["landmarks"],
                        detected_landmarks=detected_face["landmarks"],
                    )

                    if avg_landmark_distance is not None:
                        if avg_landmark_distance < config["MAX_LANDMARK_DISTANCE"]:
                            matched_person_id = person["id"]
                            break
                else:
                    # Calculate IOU between the person's facial area and the detected facial area
                    facial_area_iou = calculate_iou(
                        box1=person["facial_area"],
                        box2=detected_face["facial_area"],
                    )

                    if facial_area_iou > max_facial_area_iou:
                        max_facial_area_iou = facial_area_iou
                        matched_person_id = person["id"]

                        # Secondary check using landmarks to refine the match
                        if matched_person_id is not None:

                            avg_landmark_distance = calculate_avg_landmark_distance(
                                person_landmarks=person["landmarks"],
                                detected_landmarks=detected_face["landmarks"],
                            )

                            if avg_landmark_distance is not None:
                                if (
                                    avg_landmark_distance
                                    < config["MAX_LANDMARK_DISTANCE"]
                                    and avg_landmark_distance < min_landmark_distance
                                ):
                                    min_landmark_distance = avg_landmark_distance
                                else:
                                    matched_person_id = None

            if matched_person_id is not None:

                # Remove the pixel gap between the facial_area and the frame border introduced by RetinaFace
                remove_frame_border_gap(
                    detected_face=detected_face,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )

                # Save the person's id in the face dict
                detected_face["person_id"] = matched_person_id

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

                if tracked_persons[person_id]["facial_area"] is not None:
                    detected_faces[f"previous_{index}"] = {
                        "score": config["MAX_CONFIDENCE_SCORE"],
                        "facial_area": tracked_persons[person_id]["facial_area"],
                        "landmarks": tracked_persons[person_id]["landmarks"],
                        "person_id": person_id,
                    }

        if len(detected_faces) > 0:
            # Blur each detected face
            for detected_face in detected_faces.values():
                x1, y1, x2, y2 = detected_face["facial_area"]

                # Blur the detected face region
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(
                    src=frame[y1:y2, x1:x2],
                    ksize=(51, 51),
                    sigmaX=30,
                )

                # Print the person id in bold and white font
                person_id = detected_face["person_id"]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                color = (255, 255, 255)  # White color
                text_size, _ = cv2.getTextSize(
                    str(person_id), font, font_scale, thickness
                )
                text_x = x1 + (x2 - x1 - text_size[0]) // 2
                text_y = y1 + (y2 - y1 + text_size[1]) // 2
                cv2.putText(
                    img=frame,
                    text=f"{person_id}",
                    org=(text_x, text_y),
                    fontFace=font,
                    fontScale=font_scale,
                    color=color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )

        # Revert the color space back to RGB8
        frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

        return frame

    else:
        return None


def remove_frame_border_gap(
    detected_face: dict,
    frame_width: int,
    frame_height: int,
) -> None:
    # Ensure the facial area touches the frame border
    for landmark in detected_face["landmarks"].values():
        x, y = landmark
        if x < 0:
            detected_face["facial_area"][0] = 0
        elif x > frame_width:
            detected_face["facial_area"][2] = frame_width
        if y < 0:
            detected_face["facial_area"][1] = 0
        elif y > frame_height:
            detected_face["facial_area"][3] = frame_height

    if detected_face["facial_area"][0] == 1:
        detected_face["facial_area"][0] = 0
    if detected_face["facial_area"][1] == 1:
        detected_face["facial_area"][1] = 0
    if detected_face["facial_area"][2] == frame_width - 1:
        detected_face["facial_area"][2] = frame_width
    if detected_face["facial_area"][3] == frame_height - 1:
        detected_face["facial_area"][3] = frame_height


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
            }
        )


def update_tracked_persons(
    tracked_persons: List[dict],
    headcount_change: dict = None,
    detected_faces: dict = None,
) -> None:
    # If the face data of entering or leaving persons has to be updated
    if headcount_change is not None:
        # Identifier of the person involved in the headcount change
        person_id = headcount_change["person"]["id"]
        # If the person entered the scene
        if headcount_change["type"] == "enters":
            # Person's facial landmarks that were manually obtained
            manual_landmarks = headcount_change["person"]["landmarks"]
            # Add the manually obtained landmarks to the person's data
            for landmark, coordinates in manual_landmarks.items():
                tracked_persons[person_id]["landmarks"][landmark] = coordinates
        # If the person left the scene
        elif headcount_change["type"] == "leaves":
            # Clear their facial_area
            tracked_persons[person_id]["facial_area"] = None
            # Clear their face landmarks
            old_landmarks = tracked_persons[person_id]["landmarks"]
            for landmark, coordinates in old_landmarks.items():
                tracked_persons[person_id]["landmarks"][landmark] = None

    # If the face data of detected persons has to be updated
    elif detected_faces is not None:
        for face_id, detected_face in detected_faces.items():
            # Identifier of the person whose face was detected
            person_id = detected_face["person_id"]
            # Update the facial area and landmarks for the detected person
            tracked_persons[person_id]["facial_area"] = detected_face["facial_area"]
            for landmark, coordinates in detected_face["landmarks"].items():
                tracked_persons[person_id]["landmarks"][landmark] = coordinates


def load_config_file(filename: str) -> dict:

    try:
        # Load configuration from the YAML file
        with open(filename, "r") as config_file:
            config = yaml.safe_load(config_file)

        # Check if the config is None or empty
        if config is None or not config:
            print(
                f"\nError: The config file '{filename}' was loaded, but it contains no data."
            )
        else:
            return config

    except FileNotFoundError:
        print(f"\nError: The config file '{filename}' was not found.")
    except yaml.YAMLError as e:
        print(
            f"\nError: An issue occurred while parsing the config file '{filename}'.\n\tDetails: {e}"
        )
    except Exception as e:
        print(
            f"\nError: An issue occurred while loading the config file '{filename}'.\n\tDetails: {e}"
        )

    # Exit the program
    print(f"\nExiting the program.")
    sys.exit()


def load_metadata_file(filename: str) -> DataFrame:

    try:
        # Load the Excel file containing the dataset metadata
        with pd.ExcelFile("dataset_metadata.xlsx") as xlsx:
            dataset_metadata = pd.read_excel(xlsx)

        # Check if the DataFrame is empty
        if dataset_metadata.empty:
            print(
                f"\nError: The metadata file '{filename}' was loaded, but it contains no data."
            )
        else:
            return dataset_metadata

    except FileNotFoundError:
        print(f"\nError: The metadata file '{filename}' was not found.")
    except Exception as e:
        print(
            f"\nError: An issue occurred while loading the metadata file '{filename}'.\n\tDetails: {e}"
        )

    # Exit the program
    print(f"\nExiting the program.")
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
        print(f"\nExiting the program.")
        sys.exit()
    else:
        print(f"\nSelected dataset folder: {dataset_path}")
        print("\n------------------------------------------------------------")

    # Create the 'edits' folder if it doesn't exist
    output_directory_path = dataset_path / "edits"
    if not output_directory_path.exists():
        output_directory_path.mkdir(parents=True)

    # List all .bag files in the directory
    bag_file_paths = [
        bag_file_path
        for bag_file_path in dataset_path.glob("*.bag")
        if bag_file_path.is_file()
    ]

    # Temporary benchmark for performance measurement
    start_time = time.time()

    for bag_file_path in bag_file_paths:
        # Extract the file name from the selected path
        bag_file_name = bag_file_path.name

        print(f"\nProcessing of the file '{bag_file_name}' has started.")

        bag_metadata = dataset_metadata[
            dataset_metadata["bag_filename"] == bag_file_name
        ]

        process_bag_file(
            config=config,
            input_bag_path=bag_file_path,
            input_bag_metadata=bag_metadata,
            output_directory_path=output_directory_path,
        )

    # Calculate and display the duration of the process
    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f"\nTime: {format(round(duration, 3))} minutes")


def process_bag_file(
    config: dict,
    input_bag_path: Path,
    input_bag_metadata: DataFrame,
    output_directory_path: Path,
):

    # Open the input ROS bag file
    input_bag = rosbag.Bag(input_bag_path, "r")

    # Get the data and info topics of the color stream
    # . Initialize topic variables
    color_data_topic = None
    color_info_topic = None

    # . Obtain the list of all topics in the bag file
    input_bag_topics = input_bag.get_type_and_topic_info().topics.keys()

    # . Find the data and info ones
    for topic in input_bag_topics:
        if topic.endswith("Color_0/info"):
            color_info_topic = topic
        elif "Color_0/image/data" in topic:
            color_data_topic = topic

        # If both topics are found, no need to continue the loop
        if color_data_topic and color_info_topic:
            break

    # Get the fps, the first and last frame numbers of the color stream
    # . Initialize
    color_stream_fps = None
    first_frame_number = None
    last_frame_number = None

    # . Find the frame numbers and the fps
    for topic, msg, _ in input_bag.read_messages(
        topics=[color_data_topic, color_info_topic]
    ):
        if topic == color_info_topic:
            # Save the color stream fps
            color_stream_fps = msg.fps

        elif color_data_topic in topic:
            # Save the first frame number
            if first_frame_number is None:
                first_frame_number = msg.header.seq

            # Continuously update the last frame number
            last_frame_number = msg.header.seq

    # Minimum number of frames in an output bag
    min_output_bag_frames = color_stream_fps * config["MIN_OUTPUT_DURATION"]

    # Determine undesired intervals
    undesired_intervals = get_undesired_intervals(
        bag_metadata=input_bag_metadata,
        config=config,
        min_output_bag_frames=min_output_bag_frames,
    )

    # Determine desired intervals
    desired_intervals = get_desired_intervals(
        undesired_intervals=undesired_intervals,
        first_frame_number=first_frame_number,
        last_frame_number=last_frame_number,
        min_output_bag_frames=min_output_bag_frames,
    )
    print("\nThe desired frame intervals are:")
    for interval in desired_intervals:
        print(f"\t{interval}")

    if len(desired_intervals) > 0:

        # Initialize data structures for color stream processing
        tracked_persons = []  # Updated data of the persons appearing
        headcount_changes = []  # List of the headcount changes during the stream
        headcount_segments = []  # Stream segmented by headcount changes

        # Get the current timestamp and format it
        run_timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

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

        undesired_interval_index = None
        current_undesired_interval = None
        if len(undesired_intervals) > 0:
            undesired_interval_index = 0
            current_undesired_interval = undesired_intervals[undesired_interval_index]

        writing_desired_frames = False
        bag_processing_finished = False
        clip_start_time = None
        output_bag = None
        current_output_bag_path = None
        frame_number = None

        # Iterate through all messages in the input bag file
        for topic, msg_raw, t in input_bag.read_messages(raw=True):

            if (not bag_processing_finished) and (color_data_topic in topic):

                # Unpack the message tuple
                msg_type, serialized_bytes, md5sum, pos, pytype = msg_raw

                # Deserialize the message bytes into an Image message
                deserialized_bytes = Image()
                deserialized_bytes.deserialize(serialized_bytes)

                # Save the header values from the message
                frame_number = deserialized_bytes.header.seq
                frame_timestamp = deserialized_bytes.header.stamp

                processed_cv_image = None
                must_write_frame = False

                # Update the current desired interval
                if frame_number == current_desired_interval["start_frame"]:
                    interval = f"{current_desired_interval['start_frame']}, {current_desired_interval['end_frame']}"
                    print(
                        f"\n\nProcessing of the desired interval [{interval}] has started."
                    )
                elif (len(undesired_intervals) > 0) and (
                    frame_number == current_undesired_interval["start_frame"]
                ):
                    interval = f"{current_undesired_interval['start_frame']}, {current_undesired_interval['end_frame']}"
                    print(
                        f"\nTracking in the undesired interval [{interval}] has started."
                    )
                elif (frame_number > current_desired_interval["end_frame"]) and (
                    desired_interval_index < len(desired_intervals) - 1
                ):
                    desired_interval_index += 1
                    current_desired_interval = desired_intervals[desired_interval_index]
                elif (
                    (len(undesired_intervals) > 0)
                    and (frame_number > current_undesired_interval["end_frame"])
                    and (undesired_interval_index < len(undesired_intervals) - 1)
                ):
                    undesired_interval_index += 1
                    current_undesired_interval = undesired_intervals[
                        undesired_interval_index
                    ]

                # If the current color frame is part of a desired interval
                if is_frame_in_interval(frame_number, current_desired_interval):
                    must_write_frame = True

                if person_appearances is not None:
                    # Update the current segment of the stream if necessary
                    if frame_number > current_headcount_segment["end_frame"]:
                        headcount_segment_index += 1
                        current_headcount_segment = headcount_segments[
                            headcount_segment_index
                        ]

                    # Check if the current frame has a headcount change to update the persons
                    for headcount_change in headcount_changes:
                        if frame_number == headcount_change["frame"]:
                            # Insert or remove the facial data of the person involved in the headcount change
                            update_tracked_persons(
                                tracked_persons=tracked_persons,
                                headcount_change=headcount_change,
                            )

                    # Only process the current message if persons appear in the stream segment
                    if len(current_headcount_segment["headcount"]) != 0:
                        # Convert the ROS Image message to an OpenCV image
                        cv_image = bridge.imgmsg_to_cv2(
                            img_msg=deserialized_bytes,
                            desired_encoding="bgr8",
                        )

                        # Process the image to keep track of the persons appearing and blur their faces
                        processed_cv_image = analyze_and_anonymize_frame(
                            frame=cv_image,
                            headcount_segment=current_headcount_segment,
                            tracked_persons=tracked_persons,
                            config=config,
                            should_anonymize=must_write_frame,
                        )

                if must_write_frame:

                    if person_appearances is not None:
                        # Only process the current message if persons appear in the stream segment
                        if len(current_headcount_segment["headcount"]) != 0:
                            # Convert the OpenCV image back to a ROS Image message
                            processed_ros_image = bridge.cv2_to_imgmsg(
                                cvim=processed_cv_image,
                                encoding="rgb8",
                            )

                            # Restore the message header values. Skip restoring the frame_id to prevent issues with newer SDK versions
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
                        print(f"Created bag file: {current_output_bag_path.name}")

                        # Write the RealSense config messages to the output bag clip
                        for rs_topic, rs_msg, rs_t in input_bag.read_messages(
                            end_time=BAG_INITIALIZATION_TIME, raw=True
                        ):
                            output_bag.write(
                                topic=rs_topic, msg=rs_msg, t=rs_t, raw=True
                            )

                print(f"\rCurrent color frame: {frame_number}", end="", flush=True)

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

                    # Update the flag to disable the writing
                    writing_desired_frames = False

                    # Close the current output bag clip
                    close_bag(bag=output_bag, bag_path=current_output_bag_path)

                    # If all the desired intervals were written the task is finished
                    if desired_interval_index == len(desired_intervals) - 1:
                        bag_processing_finished = True

                    # Reset for the next clip
                    clip_start_time = None

        # If clip was being written after all the input_bag's messages were read
        if writing_desired_frames:
            # Close the current output bag clip
            close_bag(bag=output_bag, bag_path=current_output_bag_path)

    # Close the input bag file
    input_bag.close()
    print(f"\nProcessing of the file '{input_bag_path.name}' has finished.")
    print("\n------------------------------------------------------------")


def close_bag(bag: rosbag.Bag, bag_path: Path) -> None:

    bag.close()
    print(f"\nClosed bag file: {bag_path.name}")


if __name__ == "__main__":
    main()
