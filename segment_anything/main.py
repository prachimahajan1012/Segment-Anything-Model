import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import os
from PIL import Image

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#//print(device)

model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint="C:/Users/pmn27/Downloads/sam_vit_h_4b8939.pth")
sam.to(device=device)
predictor = SamPredictor(sam)
#select image
image_path ='C:/Users/pmn27/Downloads/01.JPG'
#image_path = "C:/Users/pmn27/segment-anything/images/chair_srgan.png"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

#--------------------------------------------------
# Load the image using Pillow
image_ = Image.open(image_path)
image_ = np.array(image_)

# Global variable to store the coordinates
selected_coords = []

# Function to handle mouse movements
def on_hover(event):
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        

# Function to handle mouse clicks
def on_click(event):
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        selected_coords.clear()
        selected_coords.append((x, y))
        print(f"Selected pixel: {x}, {y}")
        
        # Plot the selected point
        ax.scatter(x, y, color='green', marker='.', s=100, edgecolor='white', linewidth=1.25)
        fig.canvas.draw()

# Create the plot
fig, ax = plt.subplots()
ax.axis("off")
ax.imshow(image_)

# Connect the event handlers
fig.canvas.mpl_connect('motion_notify_event', on_hover)
fig.canvas.mpl_connect('button_press_event', on_click)


plt.show()

# After the plot is closed, you can check the selected coordinates
print(f"Selected coordinates: {selected_coords}")

if selected_coords:
    # Convert the selected coordinates to a numpy array
    input_point = np.array([selected_coords[0]])
    print(f"Selected x: {input_point[0, 0]}, y: {input_point[0, 1]}")
    print(input_point)
else:
    print("No coordinates selected.")


#--------------------------------------------------

#input_point = np.array([[1945, 525]])
print(input_point)
#box = np.array([144, 64, 666, 360])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    #box=box,
    point_labels=input_label,
    multimask_output=True,
)
print(masks.shape)

# Get the third mask
mask_to_save = masks[2]

# Create a new image with only the mask pixels
output_image = np.zeros_like(image)
output_image[mask_to_save != 0] = image[mask_to_save != 0]

# Save the image with transparency
cv2.imwrite("C:/Users/pmn27/segment-anything/masks/mask3.png", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
alpha_img = cv2.imread("C:/Users/pmn27/segment-anything/masks/mask3.png", cv2.IMREAD_UNCHANGED)

img = Image.open('C:/Users/pmn27/segment-anything/masks/mask3.png')
rgba = img.convert("RGBA") 
datas = rgba.getdata() 
  
newData = [] 
for item in datas: 
    if item[0] == 0 and item[1] == 0 and item[2] == 0:  # finding black colour by its RGB value 
        # storing a transparent value when we find a black colour 
        newData.append((255, 255, 255, 0)) 
    else: 
        newData.append(item)  # other colours remain unchanged 
  
rgba.putdata(newData) 
rgba.save("C:/Users/pmn27/segment-anything/Transparent_img/transparent_image.png", "PNG")



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


fig, axs = plt.subplots(1, 3, figsize=(15, 5))


for i, (mask, score, ax) in enumerate(zip(masks, scores, axs)):
    ax.imshow(image)

    show_mask(mask, ax)
    show_points(input_point, input_label, ax)
    #show_box(box, plt.gca())
    ax.set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show() 


T_image_path ="C:/Users/pmn27/segment-anything/Transparent_img/transparent_image.png"
image_T = Image.open(T_image_path)
image_T = np.array(image_T)
fig, ax = plt.subplots()
plt.title("Mask of Selected Object", fontdict={'fontsize':10})
ax.imshow(image_T)
ax.axis("off")
plt.show()