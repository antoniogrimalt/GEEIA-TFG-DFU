import tkinter
from tkinter import filedialog
from pathlib import Path
from datetime import datetime
from typing import Tuple, List
from sensor_msgs.msg import Image
import rosbag
import cv2
from cv_bridge import CvBridge
from retinaface import RetinaFace
import time
import json
import numpy as np
from genpy import Time
import io

# Temporary values. These will be extracted from an Excel file.
# On frame wound visibility
WOUND_VISIBILITY_FULL_CELL = '[{"start_frame":96,"end_frame":183},{"start_frame":197,"end_frame":208},{"start_frame":223,"end_frame":300},{"start_frame":386,"end_frame":420},{"start_frame":438,"end_frame":442},{"start_frame":470,"end_frame":515},{"start_frame":535,"end_frame":545},{"start_frame":588,"end_frame":626},{"start_frame":637,"end_frame":742},{"start_frame":824,"end_frame":862},{"start_frame":887,"end_frame":941},{"start_frame":968,"end_frame":1028},{"start_frame":1134,"end_frame":1140}]'
WOUND_VISIBILITY_PARTIAL_CELL = '[{"start_frame":185,"end_frame":188,"degree":1},{"start_frame":189,"end_frame":193,"degree":2},{"start_frame":194,"end_frame":196,"degree":1},{"start_frame":209,"end_frame":222,"degree":1},{"start_frame":301,"end_frame":303,"degree":1},{"start_frame":304,"end_frame":308,"degree":2},{"start_frame":309,"end_frame":320,"degree":3},{"start_frame":321,"end_frame":328,"degree":2},{"start_frame":329,"end_frame":334,"degree":1},{"start_frame":335,"end_frame":364,"degree":2},{"start_frame":365,"end_frame":380,"degree":3},{"start_frame":381,"end_frame":384,"degree":2},{"start_frame":385,"end_frame":385,"degree":1},{"start_frame":421,"end_frame":437,"degree":1},{"start_frame":443,"end_frame":452,"degree":1},{"start_frame":453,"end_frame":466,"degree":2},{"start_frame":467,"end_frame":469,"degree":1},{"start_frame":516,"end_frame":534,"degree":1},{"start_frame":546,"end_frame":567,"degree":1},{"start_frame":568,"end_frame":577,"degree":2},{"start_frame":578,"end_frame":587,"degree":1},{"start_frame":627,"end_frame":636,"degree":1},{"start_frame":743,"end_frame":762,"degree":1},{"start_frame":763,"end_frame":810,"degree":2},{"start_frame":811,"end_frame":823,"degree":1},{"start_frame":863,"end_frame":865,"degree":1},{"start_frame":866,"end_frame":871,"degree":2},{"start_frame":872,"end_frame":886,"degree":1},{"start_frame":942,"end_frame":967,"degree":1},{"start_frame":1029,"end_frame":1033,"degree":1},{"start_frame":1034,"end_frame":1046,"degree":2},{"start_frame":1047,"end_frame":1125,"degree":3},{"start_frame":1126,"end_frame":1129,"degree":2},{"start_frame":1130,"end_frame":1133,"degree":1}]'
WOUND_VISIBILITY_NONE_CELL = (
    '[{"start_frame":39,"end_frame":83},{"start_frame":1162,"end_frame":1196}]'
)
# Issues with the wound visibility
WOUND_BLURRY_CELL = (
    '[{"start_frame":84,"end_frame":94},{"start_frame":1141,"end_frame":1161}]'
)
WOUND_COVERED_CELL = (
    '[{"start_frame":994,"end_frame":1036},{"start_frame":1156,"end_frame":1161}]'
)
WOUND_NEAR_FACE_CELL = '[{"start_frame":832,"end_frame":841}]'
# Persons that can be recognized in the color frames
RECOGNIZABLE_PERSONS_CELL = '[{"type":"STAFF","appearances":[{"start":{"frame":117,"landmarks":{"right_eye":[256,485],"nose":[234,483],"mouth_right":[240,457],"mouth_left":[215,470]}},"end":{"frame":226}}]},{"type":"PATIENT","appearances":[{"start":{"frame":796,"landmarks":{"left_eye":[666,-3],"nose":[636,2],"mouth_right":[626,14],"mouth_left":[645,25]}},"end":{"frame":841}}]}]'

# Default publish time for the RealSense headers
BAG_INITIALIZATION_TIME = Time(nsecs=1)

BAG_SENSORS_START_TIME = Time(nsecs=20000)

# File types
BAG_FILE_TYPE = ("BAG File", "*.bag")
EXCEL_FILE_TYPE = ("Excel Workbook", "*.xlsx, *.xlsm, *.xlsb, *.xls")
JSON_FILE_TYPE = ("JSON File", "*.json")

# Config
FACE_REUSE_LIMIT = 3
LANDMARK_DISTANCE_THRESHOLD = 50
MAX_CONFIDENCE_SCORE = 1.0000000000000000
IGNORE_STAFF = False  # Skip blurring the Staff
MIN_OUTPUT_DURATION = 3  # Minimum output bag duration in seconds
# Wound visibility
KEEP_WOUND_COMPLETELY_OUT = False
KEEP_WOUND_PARTIALLY_OUT = True
PARTIALLY_OUT_TOLERANCE = 1  # 1: LIGHTLY, 2: MODERATELY, 3: HEAVILY
KEEP_BLURRY_WOUND = False
KEEP_COVERED_WOUND = False
KEEP_WOUND_NEAR_FACE = False


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


def get_undesired_intervals() -> List[dict]:
    undesired_intervals = []

    # Add intervals where wound visibility is none
    if (not KEEP_WOUND_COMPLETELY_OUT) and (WOUND_VISIBILITY_NONE_CELL != ""):
        undesired_intervals += json.loads(WOUND_VISIBILITY_NONE_CELL)

    # Add intervals where wound visibility is partial based on tolerance
    if WOUND_VISIBILITY_PARTIAL_CELL != "":
        partial_intervals = json.loads(WOUND_VISIBILITY_PARTIAL_CELL)
        filtered_intervals = []

        if not KEEP_WOUND_PARTIALLY_OUT:
            filtered_intervals = partial_intervals
        else:
            filtered_intervals = [
                interval
                for interval in partial_intervals
                if interval["degree"] > PARTIALLY_OUT_TOLERANCE
            ]

        # Remove the 'degree' key from partial intervals
        for interval in filtered_intervals:
            interval.pop("degree", None)

        undesired_intervals += filtered_intervals

    if (not KEEP_BLURRY_WOUND) and (WOUND_BLURRY_CELL != ""):
        undesired_intervals += json.loads(WOUND_BLURRY_CELL)

    if (not KEEP_COVERED_WOUND) and (WOUND_COVERED_CELL != ""):
        undesired_intervals += json.loads(WOUND_COVERED_CELL)

    if (not KEEP_WOUND_NEAR_FACE) and (WOUND_NEAR_FACE_CELL != ""):
        undesired_intervals += json.loads(WOUND_NEAR_FACE_CELL)

    # Combine overlapping or adjacent undesired intervals
    return combine_undesired_intervals(undesired_intervals)


def get_desired_intervals(
    undesired_intervals: List[dict],
    first_frame_number: int,
    last_frame_number: int,
    fps: int,
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
    min_frames = fps * MIN_OUTPUT_DURATION

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


def get_file_path(file_type: Tuple[str, str]) -> str:
    main_window = tkinter.Tk()
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
    cv_image: np.ndarray, current_segment: dict, current_persons: List[dict]
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

    print(f"\nColor fps: {color_fps}")

    # FACES
    current_persons = []  # Updated data of the persons showing up in the color stream

    headcount_changes = []  # List of the headcount changes during the whole stream
    headcount_segments = []  # Color stream segmented by headcount changes

    if RECOGNIZABLE_PERSONS_CELL != "":
        recognizable_persons = json.loads(RECOGNIZABLE_PERSONS_CELL)

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
    if RECOGNIZABLE_PERSONS_CELL != "":
        # Initialize the stream segment
        headcount_segment_index = 0
        current_headcount_segment = headcount_segments[headcount_segment_index]

        # Initialize the CvBridge
        bridge = CvBridge()

    # Color stream variables
    color_msg_counter = 0
    undesired_intervals = get_undesired_intervals()
    desired_intervals = get_desired_intervals(
        undesired_intervals=undesired_intervals,
        first_frame_number=first_frame_number,
        last_frame_number=last_frame_number,
        fps=color_fps,
    )
    writing_desired_frames = False
    clip_start_time = None

    if len(desired_intervals) > 0:

        current_output_bag_clip = 0
        output_bag, current_output_bag_path = None, None

        for topic, msg_raw, t in input_bag.read_messages(raw=True):

            if color_data_topic in topic:

                # Keep track of the current color message to print the progress of the process
                color_msg_counter += 1
                print(
                    f"\rProcessed {color_msg_counter}/{color_msg_count} messages from the color stream",
                    end="",
                    flush=True,
                )

                # Unpack the message tuple
                msg_type, serialized_bytes, md5sum, pos, pytype = msg_raw

                # Deserialize the message bytes
                deserialized_bytes = Image()
                deserialized_bytes.deserialize(serialized_bytes)

                # Save the header values
                frame_number = deserialized_bytes.header.seq
                frame_timestamp = deserialized_bytes.header.stamp

                if RECOGNIZABLE_PERSONS_CELL != "":
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

                # If the current color frame is part of a desired interval
                if is_frame_in_intervals(frame_number, desired_intervals):

                    if RECOGNIZABLE_PERSONS_CELL != "":
                        # Only process the current message if persons appear in the stream segment
                        if len(current_headcount_segment["headcount"]) != 0:
                            # Convert the ROS Image message to an OpenCV image
                            cv_image = bridge.imgmsg_to_cv2(
                                img_msg=deserialized_bytes,
                                desired_encoding="bgr8",
                            )

                            # Process the image using OpenCV
                            processed_cv_image = process_cv_image(
                                cv_image=cv_image,
                                current_segment=current_headcount_segment,
                                current_persons=current_persons,
                            )

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
                            output_bag_clip=current_output_bag_clip,
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

                # If the current color frame isn't part of a desired interval
                else:
                    # and a clip was being written
                    if writing_desired_frames:
                        # Update the flag to disable the writing
                        writing_desired_frames = False

                        # Close the current output bag clip
                        output_bag.close()
                        print(f"\nClosed bag file: {current_output_bag_path}")

                        # Prepare the variables for the next clip
                        current_output_bag_clip += 1
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
            print(f"\nClosed bag file: {current_output_bag_path}")

    input_bag.close()
    print(f"\nProcessing complete.\nEdited bag files saved in: {output_directory_path}")

    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f"Time: {format(round(duration, 3))} minutes")


if __name__ == "__main__":
    main()
