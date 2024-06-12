import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Load the SAM model

os.environ['KMP_DUPLICATE_LIB_OK']='True'
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint="C:/Users/pmn27/Downloads/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

# Load the image
image_path = 'C:/Users/pmn27/Downloads/01.JPG'  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Convert image to numpy array
image_np = np.array(image)

# Detect objects using SAM
masks = mask_generator.generate(image_np)

# Visualize detected masks
plt.figure(figsize=(10, 10))
plt.imshow(image_np)
for mask in masks:
    plt.contour(mask['segmentation'], colors='r')
plt.title("Detected Masks")    
plt.show()

output_dir = 'C:/Users/pmn27/segment-anything/DepthMaps'
output_path = os.path.join(output_dir, "detected_masks.png")
plt.savefig(output_path)
plt.show()

# Depth estimation using MiDaS
model_type = "DPT_Large"  # MiDaS model type
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

transform = midas_transforms.default_transform

# Prepare the image for depth estimation
input_image = transform(image_np).to(device)

# Perform depth estimation
with torch.no_grad():
    prediction = midas(input_image)

# Resize depth map to original image size
depth_map = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=image_np.shape[:2],
    mode="bicubic",
    align_corners=False,
).squeeze()

depth_map = depth_map.cpu().numpy()


os.makedirs(output_dir, exist_ok=True)
depth_map_path = os.path.join(output_dir, 'depth_map.png')

# Normalize the depth map for saving as an image
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map_normalized = depth_map_normalized.astype(np.uint8)

# Save depth map using cv2
cv2.imwrite(depth_map_path, depth_map_normalized)

# Visualize the depth map
plt.figure(figsize=(10, 10))
plt.imshow(depth_map, cmap='inferno')
plt.colorbar()
plt.title("Depth Map")
plt.axis('off')
plt.show()

# Separate out the depth map values for the detected objects
object_depths = []
for mask in masks:
    object_mask = mask['segmentation']
    object_depth = depth_map[object_mask]
    object_depths.append((object_mask, object_depth.mean()))

print(object_depths)
# Find the object with the smallest mean depth value
min_depth_object = min(object_depths, key=lambda x: x[1])

# Extract the chosen object's mask and depth values
chosen_object_mask = min_depth_object[0]
chosen_object_depth = min_depth_object[1]

# Visualize the chosen object
plt.figure(figsize=(10, 10))
plt.imshow(image_np)
plt.contour(chosen_object_mask, colors='b')
plt.title(f"Chosen Object with Mean Depth: {chosen_object_depth:.2f}")
plt.show()

output_path = os.path.join(output_dir, "chosen_object_mask.png")
plt.savefig(output_path)
#save the mask of the chosen object
