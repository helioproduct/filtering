import cv2
import numpy as np
from matplotlib import pyplot as plt 
from filter_synthesis import *
from functions import *

def main():
    path_to_image = input('Path to the image to enchance: ')
    original_image = cv2.imread(path_to_image)
    new_image = cv2.imread(path_to_image)

    # create figure 
    fig = plt.figure(figsize=(10, 7)) 

    # Original image
    fig.add_subplot(1, 2, 1) 
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis('off') 
    plt.title("Original") 

    # Filtered image
    fig.add_subplot(1, 2, 2) 
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)) 
    plt.axis('off') 
    plt.title("Blur + sharpen") 

    blur_amount = 0
    sharp_amount = 0

    slider_blur = plt.Slider(ax=plt.axes([0.1, 0.1, 0.8, 0.03]), label="X", valmin=0, valmax=100, valinit=blur_amount)
    slider_sharpen = plt.Slider(ax=plt.axes([0.1, 0.05, 0.8, 0.03]), label="Y", valmin=0, valmax=100, valinit=sharp_amount)

    slider_blur.on_changed(redraw)
    slider_sharpen.on_changed(redraw)

    def redraw(val):
        # Convert to 0.00 - 0.5
        blur_filter_amount = (slider_blur.val / 100) * 0.5
        blur_filter = normalize_filter(new_filter(13, gauss_blur, blur_amount))

        # Convert to  0.1 - 0.25 
        sharp_filter_amount = (slider_sharpen.val / 100) * (0.25-0.1) + 0.1
        sharpen_filter = normalize_filter(new_filter(5, log_sharpen, sharp_amount))

        new_image = cv2.filter2D(original_image, -1, blur_filter) 
        new_image = cv2.filter2D(new_image, -1, sharpen_filter)
    
        plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)) 

        fig.canvas.draw()

    plt.show()


    file_name = input("Enter filtered image path:")
    cv2.imwrite(file_name, new_image)



if __name__ == '__main__':
    main()