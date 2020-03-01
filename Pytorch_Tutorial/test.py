from PIL import Image
import os
import numpy as np
import torch

root = 'PennFudanPed/'
imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
print(masks)

img_path = os.path.join(root, "PNGImages", imgs[3])
mask_path = os.path.join(root, "PedMasks", masks[3])
img = Image.open(img_path).convert("RGB")
mask = Image.open(mask_path)

mask = np.array(mask)
# instances are encoded as different colors
obj_ids = np.unique(mask)
# first id is the background, so remove it
obj_ids = obj_ids[1:]

# split the color-encoded mask into a set
# of binary masks

temp = obj_ids[:, None, None]

masks = mask == obj_ids[:, None, None]

x = masks[0]

y = masks[1]

print(x.shape)

num_objs = len(obj_ids)
boxes = []
for i in range(num_objs):
    pos =  np.where(masks[i])
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    boxes.append([xmin, ymin, xmax, ymax])

boxes = torch.as_tensor(boxes, dtype=torch.float32)
# there is only one class
labels = torch.ones((num_objs,), dtype=torch.int64)
masks = torch.as_tensor(masks, dtype=torch.uint8)

image_id = torch.tensor([3])

m1 = boxes[:, 3]
m2 = boxes[:, 1]
m3 = boxes[:, 2]
m4 = boxes[:, 0]

uno = boxes[:, 3] - boxes[:, 1]
due = boxes[:, 2] - boxes[:, 0]

area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])



print("")



num_objs = len(obj_ids)

# x = np.ones((3,2))
# x[0,1] = 100
# x[0,0] = 2
# x[2,1] = 4

# y = np.unique(x)

print("")