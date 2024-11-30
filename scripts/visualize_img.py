import os
import cv2
import numpy as np

for f in os.listdir('./output/R_pred/'):
    if f.endswith('.npy'):
        name = f.split('.')[0]
        img = np.load('./output/R_pred/'+f)
        if img.ndim == 3:
            if img.shape[2] == 3:
                # RGB image (H, W, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.shape[2] == 1:
                # grayscale image (H, W, 1)
                img = img[:, :, 0]

        img = np.clip(img, a_min=0, a_max=1)
        cv2.imwrite('./output/'+name + '.jpg', img * 255)
