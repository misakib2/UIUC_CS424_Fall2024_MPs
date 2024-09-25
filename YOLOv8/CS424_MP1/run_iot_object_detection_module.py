# UIUC CS 424 Machine Problems 
# Designed and developed by Md Iftekharul Islam Sakib, Ph.D. Student, CS, UIUC (miisakib@gmail.com) under the supervision of Prof. Tarek Abdelzaher, CS, UIUC (zaher@illinois.edu)

import time
import threading
from IoTObjectDetection.IoTObjectDetectionModule import *


start_time = time.time()

print("Simulation started at the real-time: ", getRealTimeInPrintFormat())

iot_object_detection_module = iot_object_detection_module(run_yolo_flag = True)

iot_object_detection_module.run()
    
iot_object_detection_module.print_history()
iot_object_detection_module.visualize_history()

end_time = time.time()

print("Simulation ended at the real-time: ", getRealTimeInPrintFormat())
print("Total Elapsed time: %fs" % (end_time - start_time))
