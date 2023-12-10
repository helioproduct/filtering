import cv2
import numpy as np
from matplotlib import pyplot as plt 
from filter_synthesis import *
from functions import *

def main():
    path_to_image = input('Path to the image to enchance: ')
    original_image = cv2.imread(path_to_image)

    fig = plt.figure(figsize=(10, 7)) 
    old_image_plot = fig.add_subplot(1, 2, 1)
    old_image_plot.set_title("original image")
    new_image_plot = fig.add_subplot(1, 2, 2)
    new_image_plot.set_title("filtered image")
    
    fig.show()

    old_image_plot.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    slider_blur = plt.Slider(ax=plt.axes([0.1, 0.05, 0.8, 0.03]), label="BLUR", valmin=1, valmax=100, valinit=1)
    slider_sharpen = plt.Slider(ax=plt.axes([0.1, 0.00, 0.8, 0.03]), label="SHARP", valmin=1, valmax=100, valinit=1)

    def filter_image():
        # Convert to 0.00 - 0.9
        blur_filter_amount = (slider_blur.val / 100) * 0.9
        blur_filter_size = round(12 * blur_filter_amount) + 3
        if blur_filter_size % 2 == 0:
            blur_filter_size += 1
        blur_filter = normalize_filter(new_filter(blur_filter_size, gauss_blur, blur_filter_amount))

        # Convert to  0.15 - 0.275 
        sharp_filter_amount = (slider_sharpen.val / 100) * (0.275-0.15) + 0.15
        sharp_filter_size = round(8 * sharp_filter_amount) + 2
        if sharp_filter_size % 2 == 0:
            sharp_filter_size += 1
        sharp_filter = normalize_filter(new_filter(sharp_filter_size, log_sharp, sharp_filter_amount))

        new_image = cv2.filter2D(original_image, -1, blur_filter) 
        new_image = cv2.filter2D(new_image, -1, sharp_filter)

        return new_image

    def redraw(val):
        new_image_plot.imshow(cv2.cvtColor(filter_image(), cv2.COLOR_BGR2RGB)) 
        fig.canvas.draw()

    redraw(0)

    slider_blur.on_changed(redraw)
    slider_sharpen.on_changed(redraw)

    input('Pause : press any key ...')
    plt.close(fig)

    file_name = input("Enter filtered image path:")
    if file_name != "":
        cv2.imwrite(file_name, filter_image())

if __name__ == '__main__':
    main()