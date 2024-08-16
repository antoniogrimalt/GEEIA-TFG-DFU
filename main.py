import tkinter
from tkinter import filedialog
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional
from sensor_msgs.msg import Image
import rosbag
import cv2
from cv_bridge import CvBridge
from retinaface import RetinaFace
import time
import json, yaml
import numpy as np
from genpy import Time
import io
import pandas as pd

# Default publish time for the RealSense headers
BAG_INITIALIZATION_TIME = Time(nsecs=1)

BAG_SENSORS_START_TIME = Time(nsecs=20000)

# File types
BAG_FILE_TYPE = ("BAG File", "*.bag")
EXCEL_FILE_TYPE = ("Excel Workbook", "*.xlsx *.xlsm *.xlsb *.xls")
JSON_FILE_TYPE = ("JSON File", "*.json")


def combine_undesired_intervals(intervals: List[dict]) -> List[dict]:
    if not intervals:
        return []

    # Sort intervals by start_frame
    intervals.sort(key=lambda x: x["start_frame"])

    combined_intervals = [intervals[0]]

    for current_interval in intervals[1:]:
        last_interval = combined_intervals[-1]

        """
        There could be dropped frames between two undesired intervals and the adjacent check (+1) could fail.
        We need to check if the frames in between are desired, and then if these are enough to make a clip,
        if not, discard them as undesired
        """
        # If the current interval overlaps or is adjacent to the last one, merge them
        if (current_interval["start_frame"] < last_interval["end_frame"]) or (
            current_interval["start_frame"] == last_interval["end_frame"] + 1
        ):
            last_interval["end_frame"] = max(
                last_interval["end_frame"], current_interval["end_frame"]
            )
        else:
            combined_intervals.append(current_interval)

    return combined_intervals


def get_undesired_intervals(bag_metadata: pd.DataFrame, config: dict) -> List[dict]:

    wound_visibility_partial = bag_metadata.iloc[:, 29].values[0]  # Column "AD"
    wound_visibility_none = bag_metadata.iloc[:, 30].values[0]  # Column "AE"
    wound_blurry = bag_metadata.iloc[:, 31].values[0]  # Column "AF"
    wound_covered = bag_metadata.iloc[:, 32].values[0]  # Column "AG"
    wound_near_face = bag_metadata.iloc[:, 33].values[0]  # Column "AH"

    undesired_intervals = []

    # Add intervals where wound visibility is partial based on tolerance
    if not pd.isna(wound_visibility_partial):
        partial_intervals = json.loads(wound_visibility_partial)

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
    if (not config["KEEP_WOUND_COMPLETELY_OUT"]) and (
        not pd.isna(wound_visibility_none)
    ):
        undesired_intervals += json.loads(wound_visibility_none)

    if (not config["KEEP_BLURRY_WOUND"]) and (not pd.isna(wound_blurry)):
        undesired_intervals += json.loads(wound_blurry)

    if (not config["KEEP_COVERED_WOUND"]) and (not pd.isna(wound_covered)):
        undesired_intervals += json.loads(wound_covered)

    if (not config["KEEP_WOUND_NEAR_FACE"]) and (not pd.isna(wound_near_face)):
        undesired_intervals += json.loads(wound_near_face)

    # Combine overlapping or adjacent undesired intervals
    return combine_undesired_intervals(undesired_intervals)


def get_desired_intervals(
    undesired_intervals: List[dict],
    first_frame_number: int,
    last_frame_number: int,
    fps: int,
    min_output_duration: int,
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

    # Calculate the minimum number of frames for the desired interval
    min_frames = fps * min_output_duration

    # Filter out desired intervals that are shorter than the minimum duration
    filtered_desired_intervals = [
        interval
        for interval in desired_intervals
        if (interval["end_frame"] - interval["start_frame"] + 1) >= min_frames
    ]

    return filtered_desired_intervals


def is_frame_in_intervals(frame: int, intervals: List[dict]) -> bool:
    for interval in intervals:
        if interval["start_frame"] <= frame <= interval["end_frame"]:
            return True

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


def get_file_path(file_type: Tuple[str, str]) -> Path:
    main_window = tkinter.Tk()
    main_window.withdraw()  # Hide the main window

    file_path = Path(filedialog.askopenfilename(filetypes=[file_type]))

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


def calculate_avg_landmark_distance(
    person_landmarks: Dict[str, List[float]], detected_landmarks: Dict[str, List[float]]
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


def process_frame(
    frame_image: np.ndarray,
    current_segment: dict,
    current_persons: List[dict],
    frame_number: int,
    config: dict,
) -> np.ndarray:
    image_height, image_width = frame_image.shape[:2]

    assigned_person_ids = set()

    # Detect faces using RetinaFace
    detected_faces = RetinaFace.detect_faces(
        img_path=frame_image, threshold=config["FACE_DETECTION_THRESHOLD"]
    )

    # If no faces were detected
    if len(detected_faces) == 0:
        # Flip the image vertically and check again for detection
        flipped_frame_image = cv2.flip(src=frame_image, flipCode=0)
        flipped_detected_faces = RetinaFace.detect_faces(
            img_path=flipped_frame_image, threshold=config["FACE_DETECTION_THRESHOLD"]
        )
        # If faces were detected on the flipped image
        if len(flipped_detected_faces) > 0:

            # Unflip all the detected faces correcting its features
            for face_id, detected_face in flipped_detected_faces.items():
                # facial_area
                facial_area = detected_face["facial_area"]
                y2 = image_height - facial_area[1]
                y1 = image_height - facial_area[3]
                facial_area[1] = y1
                facial_area[3] = y2
                # right_eye
                right_eye = detected_face["landmarks"]["left_eye"]
                right_eye[1] = image_height - right_eye[1]
                # left_eye
                left_eye = detected_face["landmarks"]["right_eye"]
                left_eye[1] = image_height - left_eye[1]
                # nose
                nose = detected_face["landmarks"]["nose"]
                nose[1] = image_height - nose[1]
                # mouth_right
                mouth_right = detected_face["landmarks"]["mouth_left"]
                mouth_right[1] = image_height - mouth_right[1]
                # mouth_left
                mouth_left = detected_face["landmarks"]["mouth_right"]
                mouth_left[1] = image_height - mouth_left[1]

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

            for person in current_persons:

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

                # Remove the 1 pixel gap between the facial_area and the frame border introduced by RetinaFace
                if detected_face["facial_area"][0] == 1:
                    detected_face["facial_area"][0] = 0  # Replace x_min
                if detected_face["facial_area"][1] == 1:
                    detected_face["facial_area"][1] = 0  # Replace y_min
                if detected_face["facial_area"][2] == image_width - 1:
                    detected_face["facial_area"][2] = image_width  # Replace x_max
                if detected_face["facial_area"][3] == image_height - 1:
                    detected_face["facial_area"][3] = image_height  # Replace y_max

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
            update_persons(
                current_persons=current_persons,
                detected_faces=detected_faces,
            )

    # Check for missing persons
    missing_person_ids = set(current_segment["headcount"]) - assigned_person_ids

    # If there are persons missing in the detection list
    if len(missing_person_ids) > 0:

        # Iterate through the missing persons to check if their previous facial data exists and can be used
        for index, person_id in enumerate(missing_person_ids, start=1):

            if current_persons[person_id]["facial_area"] is not None:
                detected_faces[f"previous_{index}"] = {
                    "score": config["MAX_CONFIDENCE_SCORE"],
                    "facial_area": current_persons[person_id]["facial_area"],
                    "landmarks": current_persons[person_id]["landmarks"],
                    "person_id": person_id,
                }

    if len(detected_faces) > 0:
        # Blur each detected face
        for detected_face in detected_faces.values():
            x1, y1, x2, y2 = detected_face["facial_area"]

            # Blur the detected face region
            frame_image[y1:y2, x1:x2] = cv2.GaussianBlur(
                src=frame_image[y1:y2, x1:x2], ksize=(51, 51), sigmaX=30
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
                img=frame_image,
                text=f"{person_id}",
                org=(text_x, text_y),
                fontFace=font,
                fontScale=font_scale,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    frame_image = cv2.cvtColor(src=frame_image, code=cv2.COLOR_BGR2RGB)

    return frame_image


def initialize_persons(
    current_persons: List[dict], known_persons: List[dict], ignore_staff: bool
) -> None:
    for index, person in enumerate(known_persons):
        if ignore_staff and person["type"] == "STAFF":
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
            }
        )


def update_persons(current_persons, headcount_change=None, detected_faces=None):
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
                current_persons[person_id]["landmarks"][landmark] = coordinates
        # If the person left the scene
        elif headcount_change["type"] == "leaves":
            # Clear their facial_area
            current_persons[person_id]["facial_area"] = None
            # Clear their face landmarks
            old_landmarks = current_persons[person_id]["landmarks"]
            for landmark, coordinates in old_landmarks.items():
                current_persons[person_id]["landmarks"][landmark] = None

    # If the face data of detected persons has to be updated
    elif detected_faces is not None:
        for face_id, detected_face in detected_faces.items():
            # Identifier of the person whose face was detected
            person_id = detected_face["person_id"]
            # Update the facial area and landmarks for the detected person
            current_persons[person_id]["facial_area"] = detected_face["facial_area"]
            for landmark, coordinates in detected_face["landmarks"].items():
                current_persons[person_id]["landmarks"][landmark] = coordinates


def main():

    # Load config
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    input_bag_path = get_file_path(file_type=BAG_FILE_TYPE)

    input_bag_name = input_bag_path.name

    dataset_metadata_path = get_file_path(file_type=EXCEL_FILE_TYPE)
    dataset_metadata_df = pd.read_excel(
        io=dataset_metadata_path, sheet_name=0, header=3
    )

    input_bag_metadata = dataset_metadata_df[
        dataset_metadata_df.iloc[:, 6] == input_bag_name  # Column "G"
    ]

    recognizable_persons_cell = input_bag_metadata.iloc[:, 34].values[0]  # Column "AI"
    recognizable_persons = None
    if not pd.isna(recognizable_persons_cell):
        recognizable_persons = json.loads(recognizable_persons_cell)

    # Get the current timestamp and format it
    run_timestamp = datetime.now().strftime(format="%Y%m%dT%H%M%S")

    # Temporary benchmark
    start_time = time.time()

    # Get the directory part of the path
    input_directory_path = input_bag_path.parent

    # Add the new folder 'edits'
    output_directory_path = input_directory_path / "edits"

    if not output_directory_path.exists():
        output_directory_path.mkdir(parents=True)

    input_bag = rosbag.Bag(input_bag_path, "r")

    color_data_topic = None
    color_info_topic = None
    # Get the bag topics
    input_bag_info = input_bag.get_type_and_topic_info()
    input_bag_topics = input_bag_info.topics
    # Initialize an empty list to store the continuous topics
    continuous_topics = []
    # Iterate through each topic and its info
    for topic, info in input_bag_topics.items():
        # Check if topic is color info, save it in case it is
        if (color_info_topic is None) and (topic.endswith("Color_0/info")):
            color_info_topic = topic

        # Check if the topic has a frequency different from None
        if info.frequency is not None:
            # Check if topic is color data, save it in case it is
            if (color_data_topic is None) and ("Color_0/image/data" in topic):
                color_data_topic = topic

            # Add the topic to the continuous topics list
            continuous_topics.append(topic)
            # Check for corresponding metadata topics and add them
            metadata_topic = topic.replace("/data", "/metadata")
            if metadata_topic in input_bag_topics:
                continuous_topics.append(metadata_topic)

    # Get the color stream info
    color_fps = None
    first_frame_number, last_frame_number = None, None
    color_msg_count = 0
    for topic, msg, _ in input_bag.read_messages(
        topics=[color_data_topic, color_info_topic]
    ):
        if color_info_topic in topic:
            # Save the color stream fps
            color_fps = msg.fps

        elif color_data_topic in topic:
            # Increase color message count
            color_msg_count += 1

            # If it's the first message, save its frame number
            if color_msg_count == 1:
                first_frame_number = msg.header.seq

            # Remember the last frame number
            last_frame_number = msg.header.seq

    undesired_intervals = get_undesired_intervals(
        bag_metadata=input_bag_metadata, config=config
    )
    desired_intervals = get_desired_intervals(
        undesired_intervals=undesired_intervals,
        first_frame_number=first_frame_number,
        last_frame_number=last_frame_number,
        fps=color_fps,
        min_output_duration=config["MIN_OUTPUT_DURATION"],
    )
    print(f"\nUndesired intervals: {undesired_intervals}")
    print(f"Desired intervals: {desired_intervals}\n")

    if len(desired_intervals) > 0:

        # Color stream
        current_persons = []  # Updated data of the persons appearing
        headcount_changes = []  # List of the headcount changes during the stream
        headcount_segments = []  # Stream segmented by headcount changes

        if recognizable_persons is not None:

            # Initialize the list of persons
            initialize_persons(
                current_persons=current_persons,
                known_persons=recognizable_persons,
                ignore_staff=config["IGNORE_STAFF"],
            )

            # Populate the headcount changes list
            for index, person in enumerate(recognizable_persons):
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
            current_frame_number = (
                first_frame_number  # Current color stream frame number
            )
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
        if recognizable_persons is not None:
            # Initialize the stream segment
            headcount_segment_index = 0
            current_headcount_segment = headcount_segments[headcount_segment_index]

            # Initialize the CvBridge
            bridge = CvBridge()

        color_msg_counter = 0
        writing_desired_frames = False
        task_finished = False
        clip_start_time = None
        written_intervals = 0
        output_bag, current_output_bag_path = None, None

        for topic, msg_raw, t in input_bag.read_messages(raw=True):

            if (not task_finished) and (color_data_topic in topic):

                # Unpack the message tuple
                msg_type, serialized_bytes, md5sum, pos, pytype = msg_raw

                # Deserialize the message bytes
                deserialized_bytes = Image()
                deserialized_bytes.deserialize(serialized_bytes)

                # Save the header values
                frame_number = deserialized_bytes.header.seq
                frame_timestamp = deserialized_bytes.header.stamp

                # Keep track of the current color message to print the progress of the process
                color_msg_counter += 1
                print(
                    f"\rProcessed {color_msg_counter}/{color_msg_count} color data messages. Current color frame: {frame_number}",
                    end="",
                    flush=True,
                )

                processed_cv_image = None

                if recognizable_persons is not None:
                    # Update the current segment of the stream
                    if frame_number > current_headcount_segment["end_frame"]:
                        headcount_segment_index += 1
                        current_headcount_segment = headcount_segments[
                            headcount_segment_index
                        ]

                    # Check if the current frame has a headcount change to update the persons
                    for headcount_change in headcount_changes:
                        if frame_number == headcount_change["frame"]:
                            # Insert or remove the facial data of the person involved in the headcount change
                            update_persons(
                                current_persons=current_persons,
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
                        processed_cv_image = process_frame(
                            frame_image=cv_image,
                            current_segment=current_headcount_segment,
                            current_persons=current_persons,
                            frame_number=frame_number,
                            config=config,
                        )

                # If the current color frame is part of a desired interval
                if is_frame_in_intervals(frame_number, desired_intervals):

                    if recognizable_persons is not None:
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
                            output_bag_clip=written_intervals,
                            output_directory_path=output_directory_path,
                        )
                        print(f"\nCreated bag file: {current_output_bag_path}")

                        # Write the RealSense config messages to the output bag clip
                        for rs_topic, rs_msg, rs_t in input_bag.read_messages(
                            end_time=BAG_INITIALIZATION_TIME, raw=True
                        ):
                            output_bag.write(
                                topic=rs_topic, msg=rs_msg, t=rs_t, raw=True
                            )

                        # Add to the count of written intervals
                        written_intervals += 1

                # If the current color frame isn't part of a desired interval
                else:
                    # and a clip was being written
                    if writing_desired_frames:
                        # Update the flag to disable the writing
                        writing_desired_frames = False

                        # Close the current output bag clip
                        output_bag.close()
                        print(f"\nClosed bag file: {current_output_bag_path}\n")

                        # If all the desired intervals were written the task is finished
                        if written_intervals == len(desired_intervals):
                            task_finished = True

                        # Reset for the next clip
                        clip_start_time = None

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

        # If clip was being written after all the input_bag's messages were read
        if writing_desired_frames:
            # Close the current output bag clip
            output_bag.close()
            print(f"\nClosed bag file: {current_output_bag_path}\n")

    input_bag.close()
    print(f"\nProcessing complete.\nEdited bag files saved in: {output_directory_path}")

    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f"\nTime: {format(round(duration, 3))} minutes")


if __name__ == "__main__":
    main()
