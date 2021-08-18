import cv2
import PIL.Image
import os
from tqdm.notebook import tqdm

class utils:
    
    @classmethod
    def differenceHash(self, imgPath, hashSize = 8):
    
        # Read the image
        image = cv2.imread(imgPath)
        
        # Convert the given image into a grayscale image
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to a hashSize x hashSize + 1 size
        resized = cv2.resize(grayImage, (hashSize + 1, hashSize))

        # Compute horizontal gradient between adjacent pixels
        delta = resized[:, 1:] > resized [:, :-1]

        # Compute the hash by flattening the above comparison
        # And adding 2^i for the places where the col difference is high
        return sum([2 ** i for (i, v) in enumerate(delta.flatten()) if v])
    
    @classmethod
    def getHashes(self, imgDir):
        # Define a dictionary for storing hashes and it's corresponding images
        hashes = {}
        
        # Create a container to hold the images which couldn't be processed
        errorImages = []

        # Compute the hashes and add them to your dictionary
        for pth in tqdm(os.listdir(imgDir), desc = "Computing hashes..."):

            try:
                # Append the basepath to the filename
                imPath = os.path.join(imgDir, pth)  

                # Compute the imagehash
                hashValue = self.differenceHash(imPath)

                # Add the path to the hashKey if it is duplicate and create a new key if it is not
                if hashValue in hashes:
                    hashes[hashValue].append(pth)
                else:
                    hashes[hashValue] = [pth]
            except Exception as e:
                errorImages.append(pth)
                pass
            
        return (hashes, errorImages)
    