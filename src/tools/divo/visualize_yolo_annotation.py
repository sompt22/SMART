import cv2
import numpy as np

# Define the paths to the image and annotation files
file_name = 'Circle_View3_000001'
seq = 'Circle_View3'
img_ = seq+'_000001'
dataset= "divo"
print(img_)

image_path = "/home/fatih/phd/DIVOTrack/datasets/{}/images/train/{}/img1/{}.jpg".format(dataset,seq,img_)
annotation_path = "/home/fatih/phd/DIVOTrack/datasets/{}/labels_with_ids/train/{}/img1/{}.txt".format(dataset,seq,img_)
output = "/home/fatih/phd/DIVOTrack/datasets/divo/"

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
cv2.imwrite(output + "{}_cross.jpg".format(file_name), image)
#cv2.imshow("Image with annotations", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
