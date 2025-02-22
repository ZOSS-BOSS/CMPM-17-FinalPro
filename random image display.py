import os
import random
import matplotlib.pyplot as plt
from PIL import Image

categories = []
folderpath = "C:/Users/Zane/Documents/CPMP-17-ML/CMPM-17-FirstRepo/archive/Animal Image Dataset/"
for path in os.listdir(folderpath):
    # print(path)
    categories.append(folderpath + path + "/")

for i in range(25):
    animal = random.choice(categories)
    picture = os.listdir(animal)
    picture = animal + random.choice(picture)
    subplot = plt.subplot(5,5,i+1)
    subplot.imshow(Image.open(picture))
    subplot.axis("off")

plt.tight_layout()

plt.show()


    