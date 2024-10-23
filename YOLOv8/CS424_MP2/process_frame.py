# UIUC CS 424 Machine Problems 
# Designed and developed by Md Iftekharul Islam Sakib, Ph.D. Student, CS, UIUC (miisakib@gmail.com) under the supervision of Prof. Tarek Abdelzaher, CS, UIUC (zaher@illinois.edu)

from IoTObjectDetection.IoTObjectDetectionModuleHelperFunctions import *
from IoTObjectDetection.TaskEntity import *


# read the input bounding box data from file
box_info = read_json_file('../../dataset/depth_clustering_detection_flat.json')


def process_frame(frame):
    """Process frame for scheduling.

    Process a image frame to obtain cluster boxes and corresponding scheduling parameters
    for scheduling. 

    Args:
        param1: The image frame to be processed. 

    Returns:
        A list of tasks with each task containing image_path and other necessary information. 
    """

    cluster_boxes_data = get_cluster_box_info(frame, box_info)

    task_batches = []

    # student's code here
    
    


    return task_batches
    