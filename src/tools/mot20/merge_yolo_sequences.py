import os
import cv2
import uuid
from tqdm import tqdm
import shutil


"""

def prune_caltech(input_path,output_path):
    # Create output directory structure
    output_images_path = os.path.join(output_path, "images")
    output_labels_path = os.path.join(output_path, "labels_with_ids")
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)
      
    current_sequence_path = os.path.join(input_path, "labels_with_ids")
                   
    # Loop through each label in the sequence
    label_names = os.listdir(current_sequence_path)
    for label_name in tqdm(label_names, desc="Processing labels in sequence"):
        # Get image file path and check if corresponding image file exists
        label_path = os.path.join(current_sequence_path, label_name)
        shutil.copy(label_path, output_labels_path + '/' + label_name)
        image_path = os.path.join(input_path, "images" ,label_name.replace(".txt", ".png"))
        label_name = label_name.split('.')[0]
        
        image = cv2.imread(image_path)
        output_image_path = os.path.join(output_images_path, label_name + '.jpg')
        cv2.imwrite(output_image_path, image)







def merge_sequences_mot17(input_path, output_path):
    # Create output directory structure
    output_images_path = os.path.join(output_path, "images")
    output_labels_path = os.path.join(output_path, "labels_with_ids")
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    # Initialize variables
    sequence_count = 0
    max_track_id = 0

    # Loop through each sequence
    sequence_names = os.listdir(os.path.join(input_path, "labels_with_ids", "train"))
    for sequence_name in tqdm(sequence_names, desc="Processing sequences"):
        current_sequence_path = os.path.join(input_path, "labels_with_ids", "train", sequence_name, "img1")
        
        max_track_id_seq = 0          
        # Loop through each label in the sequence
        label_names = os.listdir(current_sequence_path)
        for label_name in tqdm(label_names, desc="Processing labels in sequence"):
            # Get image file path and check if corresponding image file exists
            label_path = os.path.join(current_sequence_path, label_name)
            if os.path.getsize(label_path) == 0:
                continue            
            image_path = os.path.join(input_path, "images", "train", sequence_name, "img1" ,label_name.replace(".txt", ".jpg"))
            if not os.path.exists(image_path):
                continue

            # Load image and get dimensions
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            
            if height <= 0:
                continue
            if width <= 0:
                continue

            # Generate unique image name
            unique_image_name = str(uuid.uuid4()).replace("-", "") + ".jpg"

            # Load annotations and loop through each object
            with open(label_path, "r") as f:
                annotations = f.readlines()
            for annotation in annotations:
                # Parse object information
                class_id, track_id, xcenter, ycenter, bbox_width, bbox_height = annotation.split()
                track_id = int(track_id)

                # Clip bounding box coordinates if they are outside the image
                bbox_abs_xcenter = float(xcenter)*width
                bbox_abs_ycenter = float(ycenter)*height
                bbox_abs_width = float(bbox_width)*width
                bbox_abs_height = float(bbox_height)*height
                              
                bbox_xmin = float(bbox_abs_xcenter) - float(bbox_abs_width) / 2    # int(float(xcenter) - float(bbox_width) / 2)
                bbox_ymin = float(bbox_abs_ycenter) - float(bbox_abs_height) / 2   #int(float(ycenter) - float(bbox_height) / 2)
                bbox_xmax = float(bbox_abs_xcenter) + float(bbox_abs_width) / 2    #int(float(xcenter) + float(bbox_width) / 2)
                bbox_ymax = float(bbox_abs_ycenter) + float(bbox_abs_height) / 2   #int(float(ycenter) + float(bbox_height) / 2)
                
                bbox_xmin = max(bbox_xmin, 0)
                bbox_ymin = max(bbox_ymin, 0)
                bbox_xmax = min(bbox_xmax, width - 1)
                bbox_ymax = min(bbox_ymax, height - 1)

                # Update max track id
                if track_id > max_track_id_seq:
                    max_track_id_seq = track_id

                if track_id == -1:
                    final_track_id = -1
                else:
                    final_track_id = track_id + max_track_id
                               
                # Write object information to output label file
                with open(os.path.join(output_labels_path, unique_image_name.replace(".jpg", ".txt")), "a") as f:
                    f.write(f"{class_id} {final_track_id} {((bbox_xmax - bbox_xmin) / 2 + bbox_xmin) / width} {((bbox_ymax - bbox_ymin) / 2 + bbox_ymin) / height} {(bbox_xmax - bbox_xmin) / width} {(bbox_ymax - bbox_ymin) / height}\n")

            # Copy image to output directory with unique name
            output_image_path = os.path.join(output_images_path, unique_image_name)
            cv2.imwrite(output_image_path, image)

        max_track_id += max_track_id_seq
        sequence_count += 1

    print(f"Maximum unique track id: {max_track_id}")
    




def merge_sequences_ethz(input_path, output_path):
    # Create output directory structure
    output_images_path = os.path.join(output_path, "images")
    output_labels_path = os.path.join(output_path, "labels_with_ids")
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    # Initialize variables
    sequence_count = 0
    max_track_id = 0

    # Loop through each sequence
    sequence_names = os.listdir(os.path.join(input_path))
    for sequence_name in tqdm(sequence_names, desc="Processing sequences"):
        current_sequence_path = os.path.join(input_path, sequence_name)
        
        max_track_id_seq = 0          
        # Loop through each label in the sequence
        label_names = os.listdir(os.path.join(current_sequence_path, "labels_with_ids"))      
        for label_name in tqdm(label_names, desc="Processing labels in sequence"):
            # Get image file path and check if corresponding image file exists
            label_path = os.path.join(current_sequence_path,"labels_with_ids", label_name)
            if os.path.getsize(label_path) == 0:
                continue
            image_path = os.path.join(current_sequence_path, "images" ,label_name.replace(".txt", ".png"))
            if not os.path.exists(image_path):
                continue

            # Load image and get dimensions
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            
            if height <= 0:
                continue
            if width <= 0:
                continue

            # Generate unique image name
            unique_image_name = str(uuid.uuid4()).replace("-", "") + ".jpg"

            # Load annotations and loop through each object
            with open(label_path, "r") as f:
                annotations = f.readlines()
            for annotation in annotations:
                # Parse object information
                class_id, track_id, xcenter, ycenter, bbox_width, bbox_height = annotation.split()
                track_id = int(track_id)

                # Clip bounding box coordinates if they are outside the image
                bbox_abs_xcenter = float(xcenter)*width
                bbox_abs_ycenter = float(ycenter)*height
                bbox_abs_width = float(bbox_width)*width
                bbox_abs_height = float(bbox_height)*height
                              
                bbox_xmin = float(bbox_abs_xcenter) - float(bbox_abs_width) / 2    # int(float(xcenter) - float(bbox_width) / 2)
                bbox_ymin = float(bbox_abs_ycenter) - float(bbox_abs_height) / 2   #int(float(ycenter) - float(bbox_height) / 2)
                bbox_xmax = float(bbox_abs_xcenter) + float(bbox_abs_width) / 2    #int(float(xcenter) + float(bbox_width) / 2)
                bbox_ymax = float(bbox_abs_ycenter) + float(bbox_abs_height) / 2   #int(float(ycenter) + float(bbox_height) / 2)
                
                bbox_xmin = max(bbox_xmin, 0)
                bbox_ymin = max(bbox_ymin, 0)
                bbox_xmax = min(bbox_xmax, width - 1)
                bbox_ymax = min(bbox_ymax, height - 1)

                # Update max track id
                if track_id > max_track_id_seq:
                    max_track_id_seq = track_id

                if track_id == -1:
                    final_track_id = -1
                else:
                    final_track_id = track_id + max_track_id
                               
                # Write object information to output label file
                with open(os.path.join(output_labels_path, unique_image_name.replace(".jpg", ".txt")), "a") as f:
                    f.write(f"{class_id} {final_track_id} {((bbox_xmax - bbox_xmin) / 2 + bbox_xmin) / width} {((bbox_ymax - bbox_ymin) / 2 + bbox_ymin) / height} {(bbox_xmax - bbox_xmin) / width} {(bbox_ymax - bbox_ymin) / height}\n")

            # Copy image to output directory with unique name
            output_image_path = os.path.join(output_images_path, unique_image_name)
            cv2.imwrite(output_image_path, image)

        max_track_id += max_track_id_seq
        sequence_count += 1

    print(f"Maximum unique track id: {max_track_id}")



def merge_sequences_cityscapes(input_path, output_path):
    # Create output directory structure
    output_images_path = os.path.join(output_path, "images")
    output_labels_path = os.path.join(output_path, "labels_with_ids")
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    # Loop through each sequence
    sequence_names = os.listdir(os.path.join(input_path, "labels_with_ids", "train"))
    for sequence_name in sequence_names:
        current_sequence_path = os.path.join(input_path, "labels_with_ids", "train", sequence_name)
                
        # Loop through each label in the sequence
        label_names = os.listdir(current_sequence_path)
        for label_name in label_names:
            # Get image file path and check if corresponding image file exists
            label_path = os.path.join(current_sequence_path, label_name)
            if os.path.getsize(label_path) == 0:
                continue
            image_path = os.path.join(input_path, "images", "train", sequence_name, label_name.replace(".txt", ".png"))
            if not os.path.exists(image_path):
                print("{} NOT EXITS!!".format(image_path))
                continue

            # Load image and get dimensions
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            
            if height <= 0:
                print("{} height less than or equal to zero ".format(image_path))
                continue
            if width <= 0:
                print("{} width less than or equal to zero ".format(image_path))
                continue

            # Generate unique image name
            #unique_image_name = str(uuid.uuid4()).replace("-", "") + ".jpg"

            # Load annotations and loop through each object
            with open(label_path, "r") as f:
                annotations = f.readlines()
            for annotation in annotations:
                # Parse object information
                class_id, track_id, xcenter, ycenter, bbox_width, bbox_height = annotation.split()
                track_id = int(track_id)

                # Clip bounding box coordinates if they are outside the image
                bbox_abs_xcenter = float(xcenter)*width
                bbox_abs_ycenter = float(ycenter)*height
                bbox_abs_width = float(bbox_width)*width
                bbox_abs_height = float(bbox_height)*height
                              
                bbox_xmin = float(bbox_abs_xcenter) - float(bbox_abs_width) / 2    # int(float(xcenter) - float(bbox_width) / 2)
                bbox_ymin = float(bbox_abs_ycenter) - float(bbox_abs_height) / 2   #int(float(ycenter) - float(bbox_height) / 2)
                bbox_xmax = float(bbox_abs_xcenter) + float(bbox_abs_width) / 2    #int(float(xcenter) + float(bbox_width) / 2)
                bbox_ymax = float(bbox_abs_ycenter) + float(bbox_abs_height) / 2   #int(float(ycenter) + float(bbox_height) / 2)
                
                bbox_xmin = max(bbox_xmin, 0)
                bbox_ymin = max(bbox_ymin, 0)
                bbox_xmax = min(bbox_xmax, width - 1)
                bbox_ymax = min(bbox_ymax, height - 1)
             
                               
                # Write object information to output label file
                with open(os.path.join(output_labels_path, label_name), "a") as f:
                    f.write(f"{class_id} {track_id} {((bbox_xmax - bbox_xmin) / 2 + bbox_xmin) / width} {((bbox_ymax - bbox_ymin) / 2 + bbox_ymin) / height} {(bbox_xmax - bbox_xmin) / width} {(bbox_ymax - bbox_ymin) / height}\n")

            # Copy image to output directory with unique name
            output_image_path = os.path.join(output_images_path, label_name.replace(".txt", ".jpg"))
            cv2.imwrite(output_image_path, image)

"""




def merge_sequences_data_all(input_path, output_path):
    # Create output directory structure
    output_images_path = os.path.join(output_path, "images")
    output_labels_path = os.path.join(output_path, "labels_with_ids")
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    # Initialize variables
    sequence_count = 0
    max_track_id = 0

    # Loop through each sequence
    sequence_names = os.listdir(os.path.join(input_path))
    for sequence_name in tqdm(sequence_names, desc="Processing sequences"):
        current_sequence_path = os.path.join(input_path, sequence_name)
        
        max_track_id_seq = 0          
        # Loop through each label in the sequence
        label_names = os.listdir(os.path.join(current_sequence_path, "labels_with_ids"))      
        for label_name in tqdm(label_names, desc="Processing labels in sequence"):
            # Get image file path and check if corresponding image file exists
            label_path = os.path.join(current_sequence_path,"labels_with_ids", label_name)
            if os.path.getsize(label_path) == 0:
                continue
            image_path = os.path.join(current_sequence_path, "images" ,label_name.replace(".txt", ".jpg"))
            if not os.path.exists(image_path):
                continue

            # Load image and get dimensions
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            
            if height <= 0:
                continue
            if width <= 0:
                continue

            # Generate unique image name
            unique_image_name = str(uuid.uuid4()).replace("-", "") + ".jpg"

            # Load annotations and loop through each object
            with open(label_path, "r") as f:
                annotations = f.readlines()
            for annotation in annotations:
                # Parse object information
                class_id, track_id, xcenter, ycenter, bbox_width, bbox_height = annotation.split()
                track_id = int(track_id)

                # Clip bounding box coordinates if they are outside the image
                bbox_abs_xcenter = float(xcenter)*width
                bbox_abs_ycenter = float(ycenter)*height
                bbox_abs_width = float(bbox_width)*width
                bbox_abs_height = float(bbox_height)*height
                              
                bbox_xmin = float(bbox_abs_xcenter) - float(bbox_abs_width) / 2    # int(float(xcenter) - float(bbox_width) / 2)
                bbox_ymin = float(bbox_abs_ycenter) - float(bbox_abs_height) / 2   #int(float(ycenter) - float(bbox_height) / 2)
                bbox_xmax = float(bbox_abs_xcenter) + float(bbox_abs_width) / 2    #int(float(xcenter) + float(bbox_width) / 2)
                bbox_ymax = float(bbox_abs_ycenter) + float(bbox_abs_height) / 2   #int(float(ycenter) + float(bbox_height) / 2)
                
                bbox_xmin = max(bbox_xmin, 0)
                bbox_ymin = max(bbox_ymin, 0)
                bbox_xmax = min(bbox_xmax, width - 1)
                bbox_ymax = min(bbox_ymax, height - 1)

                # Update max track id
                if track_id > max_track_id_seq:
                    max_track_id_seq = track_id

                if track_id == -1:
                    final_track_id = -1
                else:
                    final_track_id = track_id + max_track_id
                               
                # Write object information to output label file
                with open(os.path.join(output_labels_path, unique_image_name.replace(".jpg", ".txt")), "a") as f:
                    f.write(f"{class_id} {final_track_id} {((bbox_xmax - bbox_xmin) / 2 + bbox_xmin) / width} {((bbox_ymax - bbox_ymin) / 2 + bbox_ymin) / height} {(bbox_xmax - bbox_xmin) / width} {(bbox_ymax - bbox_ymin) / height}\n")

            # Copy image to output directory with unique name
            output_image_path = os.path.join(output_images_path, unique_image_name)
            cv2.imwrite(output_image_path, image)

        max_track_id += max_track_id_seq
        sequence_count += 1

    print(f"Maximum unique track id: {max_track_id}")
    with open(os.path.join(output_path, 'num_unique_id.txt'), "a") as f:
        f.write(f"{max_track_id}")

input = "/home/fatih/mnt/merged_dataset"
output = "/home/fatih/mnt/data_all"

merge_sequences_data_all(input, output)