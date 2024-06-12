import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
from PIL import Image
import supervision as sv

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#//print(device)

model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint="C:/Users/pmn27/Downloads/sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
#select image
image_path ='C:/Users/pmn27/Downloads/01.JPG'
#image_path = "C:/Users/pmn27/segment-anything/images/chair_srgan.png"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
sam_result= mask_generator.generate(image)


print(sam_result[0].keys())

#Result Visualization
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

detections = sv.Detections.from_sam(sam_result=sam_result)

annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

sv.plot_images_grid(
    images=[image, annotated_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)


# Sort the masks based on the descending order of the area
sorted_masks = sorted(sam_result, key=lambda x: x['area'], reverse=True)

# Print the information of the first 3 masks
for i, mask in enumerate(sorted_masks[:3]):
    print(f"Mask {i+1}:")
    print(f"  Area: {mask['area']}")
    print(f"  Point Coords: {mask['point_coords']}")
    print()

masks = [
    mask['segmentation']
    for mask
    in sorted(sam_result, key=lambda x: x['area'], reverse=True)
]

# Extract the segmentation masks
top_3_masks = [mask['segmentation'] for mask in sorted_masks[:3]]

sv.plot_images_grid(
    images=top_3_masks,
    grid_size=((3, 1)),
    size=(16, 16)
)
