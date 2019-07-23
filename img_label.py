# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np

index = 1

for i in range(15600):
    
    print("NO." + str(i+1) + " is begin...")
    
    for j in range(2):
        img_path = "QR_make/datasets/NO" + str(i+1) + "_version" + str(j+1) + "_error25.png"
        img = Image.open(img_path)
        img = np.array(img).astype(int)
        """
        for row in range(480):
            for line in range(480):
                piexl = img[row][line]
                if piexl == 255:
                    img[row][line] = 1
        """
        img = Image.fromarray(img).convert('L')
        img.save("QR_make/label/NO" + str(i+1) + "_version" + str(j+1) + "_error25_label.png")
        

        img_path = "QR_make/datasets/NO" + str(i+1) + "_version" + str(j+1) + "_error30.png"
        img = Image.open(img_path)
        img = np.array(img).astype(int)
        """
        for row in range(480):
            for line in range(480):
                piexl = img[row][line]
                if piexl == 255:
                    img[row][line] = 1
        """
        img = Image.fromarray(img).convert('L')
        img.save("QR_make/label/NO" + str(i+1) + "_version" + str(j+1) + "_error30_label.png")

print("Done!")
