# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:38:48 2019

@author: DELL
"""

"""
img_path = "QR_make/datasets/NO1_version1_error25.png"
img = Image.open(img_path)
img = np.array((img)).astype(int)
print(img)
"""
img = Image.open("C:/Users/QSY/Desktop/2.png")
#img = img.resize((128, 128))
img = img.resize((480, 480),Image.ANTIALIAS)
img.save("C:/Users/QSY/Desktop/2.png")
"""
from PIL import Image
import numpy as np
import random




filename="data_set/original/"+str(x+1)+".png"
img = Image.open(filename)




    for picture in range(300):


        gamma=np.array(img)

        rows=gamma.shape[0]
        cols=gamma.shape[1]

        length=random.randint(10,90)
        width=random.randint(10,90)



        change_value1=random.randint(180,240)


        change_value2=random.randint(180,240)



        change_value3=random.randint(180,240)



        change_value4=random.randint(180,240)


        for i in range(rows):
            for j in range(cols):


                if i<=length and j<=width:
                    value1=random.randint(0,50)

                    if gamma[i][j]==0:
                        gamma[i][j]=gamma[i][j]+change_value1-value1
                #gamma[i][j]=math.log(value)/0.5
                #gamma[i][j]=math.pow(value,1.5)

                elif i<=length and j>width:
                    value2=random.randint(0,50)
                    if gamma[i][j]==255:
                        gamma[i][j]=gamma[i][j]-change_value2+value2

                elif i>length and j<width:
                    value3=random.randint(0,50)
                    if gamma[i][j]==255:
                        gamma[i][j]=gamma[i][j]-change_value3+value3

                else:
                    value4=random.randint(0,50)
                    if gamma[i][j]==0:
                        gamma[i][j]=gamma[i][j]+change_value4-value4



        gamma = Image.fromarray(gamma)

        gamma.save("data_set/data/"+str(x+1)+"/"+str(x+1)+"_"+str(picture)+".png")
        print("done! "+"number:"+str(picture))
        print("\n")


"""


