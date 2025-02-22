import matplotlib.pyplot as plt
from PIL import Image
import os
import random
# image = Image.open ("archive/Animal Image Dataset/cats/1.jpeg")
# image.show()

images = []
folderpath = "C:/Users/Zane/Documents/CPMP-17-ML/CMPM-17-FirstRepo/archive/Animal Image Dataset/cats/"
for path in os.listdir(folderpath):
    print(path)
    images.append(folderpath + path)

random.shuffle(images)
for idx, imagepath in enumerate(images):
    # if (idx < 25):
    #     continue
    subplot = plt.subplot(5,5,(idx%25)+1)
    subplot.imshow(Image.open(imagepath))
    # subplot.set_title("cat")
    subplot.axis("off")
   
    if(idx==24):
        break

plt.tight_layout()

plt.show()