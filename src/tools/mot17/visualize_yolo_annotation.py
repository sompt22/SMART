import cv2
import numpy as np

# Define the paths to the image and annotation files
file_name = 'fdea44a635e14176ab658d478aa67839'
dataset= "data_all"

image_path = "/home/fatih/mnt/{}/images/{}.jpg".format(dataset, file_name)
annotation_path = "/home/fatih/mnt/{}/labels_with_ids/{}.txt".format(dataset, file_name)
output = "/home/fatih/mnt/visual/"

# Load the image
image = cv2.imread(image_path)

# Load the annotation file and parse the bounding box coordinates
with open(annotation_path, "r") as f:
    annotations = f.readlines()
for annotation in annotations:
    class_id, track_id, x_center_norm, y_center_norm, width_norm, height_norm = map(float, annotation.split())
    height, width, _ = image.shape
    x_center = float(x_center_norm * width)
    y_center = float(y_center_norm * height)
    width = float(width_norm * width)
    height = float(height_norm * height)
    
    # Define the bounding box coordinates
    x_min = int(x_center - (width / 2))
    y_min = int(y_center - (height / 2))
    x_max = int(x_center + (width / 2))
    y_max = int(y_center + (height / 2))

    # Draw the bounding box on the image
    color = (0, 0, 255)
    thickness = 2
    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

# Show the image with the bounding box overlay
cv2.imwrite(output + "{}.jpg".format(file_name), image)
#cv2.imshow("Image with annotations", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
