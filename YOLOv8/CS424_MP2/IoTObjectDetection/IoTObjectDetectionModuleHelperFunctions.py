# UIUC CS 424 Machine Problems 
# Designed and developed by Md Iftekharul Islam Sakib, Ph.D. Student, CS, UIUC (miisakib@gmail.com) under the supervision of Prof. Tarek Abdelzaher, CS, UIUC (zaher@illinois.edu)

import cv2
import time
import os
import numpy as np
import json
import sys
from ultralytics import YOLO
from PIL import Image


def visualize_history_file(history, start_time=time.time(), Text_colors=(255,255,255)):
    """Visualize processing history from dictionary.

    Draw the processing order of bounding boxes in the processing_order_output_directory.
    Blue for box that meet deadline and red for box that missed.

    Args:
        history: A dictionary of processing history read from json file. 
    """
    history_visualization_start_time =  time.time()-start_time
    print("History visualization started at the real-time: ", getRealTimeInPrintFormat())
    
    for order in history:
        entry = history[order]

        if os.path.exists(entry["processing_order_out_path"]):
            image = cv2.imread(entry["processing_order_out_path"])
        else:
            image = cv2.imread(entry["image_path"])
            
        image_h, image_w, _ = image.shape

        if (entry["missed"]):
            bbox_color = (0,0,255)  
        else:
            bbox_color = (255,0,0)

        bbox_thick = int(0.6 * (image_h + image_w) / 1000)

        if bbox_thick < 1: 
            bbox_thick = 1

        fontScale = 0.75 * bbox_thick
        coor = entry["coord"]
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)
        order_text = "order: " + str(order)
        # get text size
        (text_width, text_height), baseline = cv2.getTextSize(order_text, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                fontScale, thickness=bbox_thick)
        # put filled text rectangle
        cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

        # put text above rectangle
        cv2.putText(image, order_text, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)
        
        i = entry["processing_order_out_path"].rfind('/')
        output_directory = entry["processing_order_out_path"][:i]
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
                
        cv2.imwrite(entry["processing_order_out_path"], image)
    
    history_visualization_end_time =  time.time()-start_time
    print("History visualization completed at the real-time: ", getRealTimeInPrintFormat())
    print("History visualization took a total time of ", history_visualization_end_time-history_visualization_start_time, "s")

    
def visualize_boxes(image_folder, ground_truth, cluster_box_info, start_time=time.time(), Text_colors=(255,255,255)):
    """Visualize processing order from dictionary.

    Draw the processing order of bounding boxes in the processing_order_output_directory.
    Blue for box that meet deadline and red for box that missed. 

    Args:
        history: A dictionary of scheduling history read from json file. 
    """
    cluster_boxes_visualization_start_time =  time.time()-start_time
    print("Cluster boxes visualization started at the real-time: ", getRealTimeInPrintFormat())
    
    for image_name in cluster_box_info:
        cluster_boxes = cluster_box_info[image_name]
        true_boxes = ground_truth[image_name]

        # get image information
        image_path = image_folder + image_name
        processing_order_output_directory = image_folder+"object_processing_order_history/"
        image_processing_order_out_path = processing_order_output_directory+image_name


        if os.path.exists(image_processing_order_out_path):
            image = cv2.imread(image_processing_order_out_path)
        else:
            image = cv2.imread(image_path) 

        image_h, image_w, _ = image.shape
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: 
            bbox_thick = 1
        fontScale = 0.75 * bbox_thick

        # draw cluster boxes
        for box in cluster_boxes:
            if len(box) < 6:
                print("scheduled_boxes.json has not been processed by get_statistics().")
                return -1

            bbox_color = (0,0,255) if (box[-1] == 0) else (255,0,0)
            (x1, y1), (x2, y2) = (box[0], box[1]), (box[2], box[3])

            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        # draw true bounding boxes
        for true_box in true_boxes:
            bbox_color = (0,255,0)
            (x1, y1), (x2, y2) = (true_box[0], true_box[1]), (true_box[2], true_box[3])

            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)


        if not os.path.exists(processing_order_output_directory):
            os.makedirs(processing_order_output_directory)

        cv2.imwrite(image_processing_order_out_path, image)

    cluster_boxes_visualization_end_time =  time.time()-start_time
    print("Cluster boxes visualization completed at the real-time: ", getRealTimeInPrintFormat())
    print("Cluster boxes visualization took a total time of ", cluster_boxes_visualization_end_time-cluster_boxes_visualization_start_time, "s")


def get_statistics_per_image(image, ground_truth, cluster_box_info):
    """Get coverage and accuracy for a single frame.
    """
    if image in ground_truth and image in cluster_box_info:
        true_boxes = ground_truth[image]
        cluster_boxes = cluster_box_info[image]
        coverage = [0] * len(true_boxes)
        cluster_statistic = [0] * len(cluster_boxes)
        pixel_array = np.zeros((len(true_boxes),1280,1920))

        i, j = 0, 0
        for entry in true_boxes:
            true_box = [entry[0], entry[1], entry[2], entry[3]]
            j = 0
            for entry2 in cluster_boxes:
                box = [int(entry2[0]), int(entry2[1]), int(entry2[2]), int(entry2[3])]
                overlap = intersection(true_box, box)
                if len(entry2) <= 5:
                    # add the sixth field
                    entry2.append(0)
                if overlap and abs(entry[4] - entry2[4]) < 10:
                    # update statistics
                    cluster_statistic[j] = 1
                    entry2[5] = 1
                    set_image_pixel_value(pixel_array[i], overlap, 1)
                j += 1
            # calculate coverage for this bounding box
            coverage[i] =  np.count_nonzero(pixel_array[i] == 1) / \
                    ((true_box[2] - true_box[0]) * (true_box[3] - true_box[1]))
            i += 1
        accuracy = sum(cluster_statistic) / len(cluster_statistic)
        return [coverage, accuracy]
    else:
        return 0


def get_statistics(image_folder, ground_truth, cluster_box_info, start_time=time.time()):
    """Get average coverage for bounding boxes and accuracy for cluster boxes.

    Process the ground truth bounding boxes and cluster box information to get 
    the average coverage for bounding boxes and accuracy for cluster boxes.
    This function also adds a sixth field to scheduled_boxes.json indicating 
    whether the box has some overlap with ground truth bounding boxes.

    Args:
        ground_truth: dictionary of Waymo ground truth bounding box.
        cluster_box_info: dictionary of Waymo ground truth bounding box.
    """
    get_statistics_start_time =  time.time()-start_time
    print("get_statistics started at the real-time: ", getRealTimeInPrintFormat())
    
    processing_order_output_directory = image_folder+"object_processing_order_history/"
    avg_coverage = []
    avg_accuracy = []
    for image in ground_truth:
        result = get_statistics_per_image(image, ground_truth, cluster_box_info)
        if result:
            avg_coverage.extend(result[0])
            avg_accuracy.append(result[1])
    coverage = sum(avg_coverage) / len(avg_coverage)
    accuracy = sum(avg_accuracy) / len(avg_accuracy)

    with open(processing_order_output_directory+'camera_frame_processed_boxes.json', 'w') as outfile:
            json.dump(cluster_box_info, outfile, ensure_ascii=False, indent=4)

    print("average coverage: %.3f" % (coverage))
    print("average accuracy: %.3f" % (accuracy))
    
    get_statistics_end_time =  time.time()-start_time
    print("get_statistics completed at the real-time: ", getRealTimeInPrintFormat())
    print("get_statistics took a total time of ", get_statistics_end_time-get_statistics_start_time, "s")



def get_group_avg_response_time(history):
    """Calculate average response time for each depth group.

    Use the processing history to calculate the average response time for 
    each depth group. Each group is composed of objects that are in a 10m
    range, such as 0-10m, 10-20m, etc..

    Args:
        history: A dictionary of processing history read from json file. 
    
    Returns:
        A list of response time for each depth group. 
        For example,
        [25.293, 31.901, 9.244, 8.324, 3.987, 1.0, 0, 1.0, 1.0, 0]
    """

    res_time = [0] * 10
    group_cnt = [0] * 10
    result = []

    for key in history:
        entry = history[key]
        group_id = int(entry["depth"] / 10)
        res_time[group_id] += entry["response_time"]
        group_cnt[group_id] += 1
    
    for i in range(10):
        if group_cnt[i] != 0:
            result.append(float("{:.3f}".format(res_time[i] / group_cnt[i])))
        else:
            result.append(0)

    return result


def get_group_worst_response_time(history):
    """Calculate worst response time for each depth group.

    Use the scheduling history to calculate the average response time for 
    each depth group. Each group is composed of objects that are in a 10m
    range, such as 0-10m, 10-20m, etc..

    Args:
        history: A dictionary of scheduling history read from json file. 
    
    Returns:
        A list of response time for each depth group. 
        For example,
        [25.293, 31.901, 9.244, 8.324, 3.987, 1.0, 0, 1.0, 1.0, 0]
    """

    res_time = [0] * 10

    for key in history:
        entry = history[key]
        group_id = int(entry["depth"] / 10)
        if entry["response_time"] > res_time[group_id]:
            res_time[group_id] = entry["response_time"]

    return res_time


def detect_images(inference_model, image_path, box=None, output_path="", id=0, write_file=True, show=False):
    """Object classification of the given image.

    Run the yolo model on the given image. With post process including nms. 
    Save the output image to file or show the image if specified. 

    Args:
        model: The yolo model to be used. 
        image_path: path to the image.
        box: bounding box coordinates. Should be a list like: [x1, y1, x2, y2].
        output_path: path to write the output image. 
        id: index of bounding box for a given frame.
        show: whether to show the image for display.
    """
    original_image = cv2.imread(image_path)

    if box:
        original_image = original_image[box[1]:box[3], box[0]:box[2]]

    # original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    
    prediction_result = inference_model.predict(original_image)
    prediction_result_image_array = prediction_result[0].plot(conf=True, img=original_image)
    prediction_result_image = Image.fromarray(prediction_result_image_array[..., ::-1])  # RGB PIL image
    
    if id:
        i = output_path.rfind('.')
        output_path = output_path[:i] + '_' + str(id) + output_path[i:]
        
    if output_path != '' and write_file: 
        i = output_path.rfind('/')
        output_directory = output_path[:i]
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        if id:
            prediction_result_image.save(output_path)
        else:
            files_with_same_prefix_in_output_directory = count_files_with_same_prefix(output_path)
            i = output_path.rfind('.')
            output_path = output_path[:i] + '[' + str(files_with_same_prefix_in_output_directory+1) + ']' +output_path[i:] 

            prediction_result_image.save(output_path)

    if show:
        # Show the image
        cv2.imshow("Predicted Image", prediction_result_image_array)
        # Load and hold the image
        cv2.waitKey(0)
        # To close the window after the required kill value was provided
        cv2.destroyAllWindows()


def load_Yolo_model():
    """Load a yolo model and its weights for inference."""
    inference_model = YOLO("yolov8n.pt")
    return inference_model


def extract_png_files(input_path):
    '''Find all png files within the given directory, sorted numerically.'''
    input_files = []
    file_names = os.listdir(input_path)

    for file in file_names:
        if ".png" in file:
            input_files.append(os.path.join(input_path, file))
    input_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return input_files


def read_json_file(filename):
    '''Return a dictionary read from json file.'''
    with open(filename) as json_file:
        data = json.load(json_file)
        return data


def get_cluster_box_info(frame, cluster_boxes):
    """Find cluster box information for the given frame.

    Get the cluster box information for the input frame from a dictionary.

    Args:
        frame: The image frame to be searched.
        cluster_boxes: a dictionary containing bounding box data.

    Returns:
        A list with the related bounding box data, including coordinates, depth, etc..
        For example, 
        [
            [644, 655, 729, 720, 64.44659992346784, ...],
            [571, 667, 759, 813, 29.452592092432084, ...],
            [1322, 764, 1920, 1214, 9.531812389460798, ...]
        ]
    """
    image_path = frame.path
    i = image_path.rfind('/')
    image_name = image_path[i+1:]

    if image_name in cluster_boxes:
        cluster_box_raw = cluster_boxes[image_name]
        cluster_box = []
        for entry in cluster_box_raw:
            tmp = []
            tmp.append(int(entry[0]))
            tmp.append(int(entry[1]))
            tmp.append(int(entry[2]))
            tmp.append(int(entry[3]))
            tmp.append(float(entry[4]))
            tmp.append(int(entry[5]))
            cluster_box.append(tmp)
        return cluster_box
    else:
        sys.exit("Error: no cluster box info for image {:s}".format(image_path))

        
def get_bbox_info(frame, cluster_boxes):
    """Find bounding box information for the given frame.

    Get the bounding box information for the input frame from a dictionary.

    Args:
        frame: The image frame to be searched.
        cluster_boxes: a dictionary containing bounding box data.

    Returns:
        A list with the related bounding box data, including coordinates, depth, etc..
        For example, 
        [
            [644, 655, 729, 720, 64.44659992346784, ...],
            [571, 667, 759, 813, 29.452592092432084, ...],
            [1322, 764, 1920, 1214, 9.531812389460798, ...]
        ]
    """
    image_path = frame.path
    i = image_path.rfind('/')
    image_name = image_path[i+1:]

    if image_name in cluster_boxes:
        cluster_box = cluster_boxes[image_name]
        return cluster_box
    else:
        sys.exit("Error: no cluster box info for image {:s}".format(image_path))


def crop_cluster_box(frame, cluster_boxes_data):
    """Crop cluster boxes from frame

    Crop a image frame into several cluster boxes as specified by the info.

    Args:
        frame: The image frame to be cropped.
        cluster_boxes_data: a dictionary containing cluster boxes data.

    Returns:
        A list with each entry having two elements. The first element is the cropped image, 
        the second element the related cluster box data, including coordinates, depth, etc..
        For example:
        [[image1, data], [image2, data], [image3, data]]
    """

    images = []
    for box in cluster_boxes_data:
        images.append([frame.image[box[1]:box[3], box[0]:box[2]], box])
    return images


def list_to_str(l):
    """Function convert a coordinate list to string for printing"""
    return '(' + str(l[0]) + ',' + str(l[1]) + '), (' + str(l[2]) + ',' + str(l[3]) + ')'


def count_files_with_same_prefix(given_file_path):
    i = given_file_path.rfind('/')
    given_directory_path = given_file_path[:i+1]
    given_file_name_with_ext = given_file_path[i+1:]
    j = given_file_name_with_ext.rfind('.')
    given_file_name = given_file_name_with_ext[:j]
    given_file_ext = given_file_name_with_ext[j+1:]

    no_of_files_with_same_prefix_found=0
    for file in os.listdir(given_directory_path):
        if file.endswith(given_file_ext) and file.startswith(given_file_name+"["):
            no_of_files_with_same_prefix_found = no_of_files_with_same_prefix_found + 1
    return no_of_files_with_same_prefix_found


def getRealTimeInPrintFormat():
    # Get the current time in seconds since the epoch
    current_time = time.time()

    # Convert the current time to a structured time format (tuple)
    structured_time = time.localtime(current_time)

    # Format the date and time
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", structured_time)
    
    return formatted_time


def line_intersection(a0, a1, b0, b1):
    """Get intersection for a line.
    """
    if a0 >= b0 and a1 <= b1: # Contained
        intersection = [a0, a1]
    elif a0 < b0 and a1 > b1: # Contains
        intersection = [b0, b1]
    elif a0 < b0 and a1 > b0: # Intersects right
        intersection = [b0, a1]
    elif a1 > b1 and a0 < b1: # Intersects left
        intersection = [a0, b1]
    else: # No intersection (either side)
        intersection = 0

    return intersection


def intersection(box1, box2):
    """Find intersection of the two boxes
    """
    inter_x = line_intersection(box1[0], box1[2], box2[0], box2[2])
    inter_y = line_intersection(box1[1], box1[3], box2[1], box2[3])

    if inter_x and inter_y:
        return [inter_x[0], inter_y[0], inter_x[1], inter_y[1]]
    else:
        return 0


def set_image_pixel_value(pixels, box, value):
    """Increase the box area of pixels with specified value
    """
    for i in range(box[1]-1, box[3]-1):
        pixels[i][box[0]-1:box[2]-1] = value