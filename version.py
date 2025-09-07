import sys
import cv2
import numpy as np
import tensorflow as tf
import cvzone
import math
import mediapipe as mp
print("mediapipe",mp.__version__)
print("cv2 version:",cv2.__version__)

print("Python version:", sys.version)
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)
print("CvZone version:", cvzone.__version__)
print(cv2.__version__)

# List all installed packages with versions (optional, for complete dependency check)
try:
    import pkg_resources
    installed_packages = pkg_resources.working_set
    for dist in installed_packages:
        print(f"{dist.project_name}=={dist.version}")
except ImportError:
    print("pkg_resources not available")
