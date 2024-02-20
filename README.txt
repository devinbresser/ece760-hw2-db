ECE 766 - HW2 - Devin Bresser

README comments:

Walkthrough 1: No comments - straightforward applications.

Challenge 1:

a.) A binarizing threshold that I found worked well for all of the provided images is 130/256. I used that threshold when binarizing the images to create the object databases in part b.)

b.) In addition to the required features, I also computed the following features:

-The ratio E_min/area for the object. The area is simply the sum of pixels equal to one in the post-labeled image.

-Coordinates of a line endpoint that I can use to draw the lines on the image. These were computed using the sine and cosine of the orientation in radians. I picked a line length of 30 pixels arbitrarily, as it looked acceptable on the output image. This method worked well to visualize the orientation angle.

c.) I only used one feature to classify the objects using the database. The feature was the ratio R = E_min/area for the object. 
This ratio was found to be a useful metric for image classification and, in fact, was able to do the job well on its own. The reason for normalizing by area is to standardize E_min against variations in object area so that an object could be scaled differently and the recognition system would still function.

