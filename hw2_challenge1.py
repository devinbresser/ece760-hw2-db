import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from PIL import Image
import numpy as np
import skimage


def generateLabeledImage(gray_img: np.ndarray, threshold: float) -> np.ndarray:
    '''
    Generates a labeled image from a grayscale image by assigning unique labels to each connected component.
    Arguments:
        gray_img: grayscale image.
        threshold: threshold for the grayscale image.
    Returns:
        labeled_img: the labeled image.
    '''
    
    ### Solution - Challenge 1a - Devin Bresser ###
    
    # 1. Binarize the image with the given threshold
    gray_img_array = np.array(gray_img)
    gray_img_array_bin = (gray_img_array > threshold)

    # 1b. Display image
    plt.imshow(gray_img_array_bin, cmap='gray')
    plt.axis('off') 
    plt.show()

    # 2. Segment the binarized image into connected regions
    labeled_img = skimage.measure.label(gray_img_array_bin, background=0, connectivity=1)

    # 2b. Display image
    plt.imshow(labeled_img, cmap='gray')
    plt.axis('off')
    plt.show()

    # 3. Annotate the labeled image with label numbers

    # Recreate labeled image
    plt.figure()
    plt.imshow(labeled_img, cmap='gray')
    plt.axis('off')
    
    # Compute object centroids (location where to put the labels)
    object_labels = np.unique(labeled_img) # extract the sorted object labels
    
    for label in object_labels:

        if label == 0:
        # Special handling for background label
        # Put it at the top right corner of the image to avoid conflicts
            plt.text(labeled_img.shape[1]-10, 10, str(label), color="red", ha="right", va="top", fontsize=26, weight='bold', clip_on=True)
            continue
            
        rows, cols = np.where(labeled_img == label) # Separate the labeled image into subregions
        centroid_row = np.mean(rows) # Compute centroid row
        centroid_col = np.mean(cols) # and column
        # Add the non-background labels to each centroid
        plt.text(centroid_col, centroid_row, str(label), color="red", ha="center", va="center", fontsize="20", weight="bold") 

    # To save as a file: 
    # plt.savefig('outputs/labeled_image.png')
    plt.show()

    return labeled_img
        
    ###
    


def compute2DProperties(orig_img: np.ndarray, labeled_img: np.ndarray) ->  np.ndarray:
    '''
    Compute the 2D properties of each object in labeled image.
    Arguments:
        orig_img: the original image.
        labeled_img: the labeled image.
    Returns:
        obj_db: the object database, where each row contains the properties
            of one object.
        out_img: the output image with the object properties annotated.
    '''
    ### Solution - Challenge 1b - Devin Bresser

    # Store width and height of image
    height, width = np.shape(orig_img)

    # Initialize object DB
    obj_db = []

    # Extract unique objects (same code as part a)
    object_labels = np.unique(labeled_img) # extract the sorted object labels

    # Loop over each item
    for label in object_labels:

        if label == 0: 
            continue # do not process background
            
        obj_info = [label] # tuple requirement 1.
        
        # Compute centroid (same code from my solution for part a)
        indices = np.argwhere(labeled_img == label) # Separate the labeled image into subregions
        rows, cols = indices[:,0], indices[:,1] # extract rows and columns of each subregion
        
        centroid_row = np.mean(rows) # Compute centroid row
        centroid_col = np.mean(cols) # and column
        obj_info.append(centroid_row) # tuple requirement 2.
        obj_info.append(centroid_col) # tuple requirement 3.

        ## Compute minimum moment of inertia
        
        # Compute a, b, c using loops
        a,b,c = 0,0,0
        
        # Compute a, b, and c per slides
        a = np.sum((cols-centroid_col)**2)
        b = 2*np.sum((rows-centroid_row)*(cols-centroid_col))
        c = np.sum((rows-centroid_row)**2)

        #print(f"a: {a}, b: {b}, c: {c}") # test

        # Compute theta_1 and theta_2 per slides
        theta_1 = 0.5*np.arctan2(b, a-c)
        theta_2 = theta_1 + 0.5*np.pi
        
        # Compute E_min and E_max per slides
        E1 = (a*np.sin(theta_1)**2 - b*np.sin(theta_1)*np.cos(theta_1) + c*np.cos(theta_1)**2)
        E2 = (a*np.sin(theta_2)**2 - b*np.sin(theta_2)*np.cos(theta_2) + c*np.cos(theta_2)**2)
        E_max = np.max([E1, E2])
        E_min = np.min([E1, E2])
    
        # Compute orientation (a little ugly code but ok)
        if(E_min == E1):
            orientation = np.rad2deg(theta_1)
        if(E_min == E2):
            orientation = np.rad2deg(theta_2)
        
        # Compute roundness
        roundness = E_min/E_max

        # Append E_min, orientation and roundness to tuple (req. 4,5,6)
        obj_info.append(E_min)
        obj_info.append(orientation)
        obj_info.append(roundness)

        # I also want to store the ratio of E_min to the area of the object (for object classification in part C)
        area = np.sum(labeled_img == label)
        obj_info.append(E_min / area)

        # One more property to store: a point 30px in the direction of orientation (for drawing lines)
        orientation_rad = np.deg2rad(orientation)
        line_endpoint_row = centroid_row + 30*np.sin(orientation_rad)
        line_endpoint_col = centroid_col + 30*np.cos(orientation_rad)
        obj_info.append(line_endpoint_row)
        obj_info.append(line_endpoint_col)


        # Append object tuple to database and repeat all of this for next object
        obj_db.append(obj_info)

    # Draw the dots and lines on the image
    fig, ax = plt.subplots()
    ax.imshow(orig_img, cmap='gray')
    
    for obj_info in obj_db:
        _, centroid_row, centroid_col, _, _, _, _, line_endpoint_row, line_endpoint_col = obj_info
        ax.plot(centroid_col, centroid_row, "ro") # red dot at centroid
        ax.plot([centroid_col, line_endpoint_col], [centroid_row, line_endpoint_row], "r-") # red line in direction of orientation

    plt.show()
    return obj_db


    ###

def recognizeObjects(orig_img: np.ndarray, labeled_img: np.ndarray, obj_db: np.ndarray, output_fn: str):
    '''
    Recognize the objects in the labeled image and save recognized objects to output_fn
    Arguments:
        orig_img: the original image.
        labeled_img: the labeled image.
        obj_db: the object database, where each row contains the properties 
            of one object.
        output_fn: filename for saving output image with the objects recognized.
    '''

    # I am only using one feature for classification. It did a good job in my experiments.
    # The feature is defined as the ratio of E_min to object area. See README for more information.

    # The methodology to identify matches will be:
        # 1. Extract objects and their features from the labeled input image using compute2DProperties
        # 2. For each object, compare R = E_min / area.
        # 3. If R_true / R_db > threshold, classify the object as a member of that class
        # See README for information on threshold.
    
    input_features = compute2DProperties(orig_img, labeled_img)
    R_input = [[feature[0], feature[6]] for feature in input_features] # Extract R from each item in the input feature
    R_db = [[object[0], object[6]] for object in obj_db] # Extract R from each item in the database

    # Now compare each input feature's value of R_input to all of those in R_db and stop if there is a match
    threshold = 0.9
    matches = []

    # Iterate over input features
    for input_feature in R_input:
        input_label, R_input_value = input_feature

        # Iterate over features in the object database
        for db_feature in R_db:
            db_label, R_db_value = db_feature
            ratio = R_input_value / R_db_value
            #print(f"ratio: {ratio}")

            # Identify a match (0.9 < ratio < 1.1)
            if ratio >= threshold and ratio <= 1/threshold:
                matches.append([input_label, db_label]) # append the match to "matches"
                break

    matched_labels = [match[0] for match in matches]
    input_features_filtered = [feature for feature in input_features if feature[0] in matched_labels]
    
    # Now that the input features are filtered to matches only, create output image
    # This is very similar to Part b so I will reuse the code
    # Draw the dots and lines on the image
    fig, ax = plt.subplots()
    ax.imshow(orig_img, cmap='gray')
    
    for input_feature in input_features_filtered:
        _, centroid_row, centroid_col, _, _, _, _, line_endpoint_row, line_endpoint_col = input_feature
        ax.plot(centroid_col, centroid_row, "go") # green dot at centroid
        ax.plot([centroid_col, line_endpoint_col], [centroid_row, line_endpoint_row], "g-") # green line in direction of orientation

    plt.axis('off')
    plt.savefig(output_fn)
    plt.show()
           
    return
    
    
    
    
def hw2_challenge1a():
    import matplotlib.cm as cm
    from skimage.color import label2rgb
    from hw2_challenge1 import generateLabeledImage
    img_list = ['two_objects.png', 'many_objects_1.png', 'many_objects_2.png']
    threshold_list = [130/256, 130/256, 130/256]   # You need to find the right thresholds

    for i in range(len(img_list)):
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.
        labeled_img = generateLabeledImage(orig_img, threshold_list[i])
        Image.fromarray(labeled_img.astype(np.uint8)).save(
            f'outputs/labeled_{img_list[i]}')
        
        cmap = np.array(cm.get_cmap('Set1').colors)
        rgb_img = label2rgb(labeled_img, colors=cmap, bg_label=0)
        Image.fromarray((rgb_img * 255).astype(np.uint8)).save(
            f'outputs/rgb_labeled_{img_list[i]}')

def hw2_challenge1b():
    labeled_two_obj = Image.open('outputs/labeled_two_objects.png')
    labeled_two_obj = np.array(labeled_two_obj)
    orig_img = Image.open('data/two_objects.png')
    orig_img = np.array(orig_img.convert('L')) / 255.
    obj_db  = compute2DProperties(orig_img, labeled_two_obj)
    np.save('outputs/obj_db.npy', obj_db)
    print(obj_db)
    
    # TODO: Plot the position and orientation of the objects
    # Use a dot or star to annotate the position and a short line segment originating from the dot for orientation
    # Refer to demoTricksFun.py for examples to draw dots and lines. 

    ### Comment: I implemented this within the function compute2DProperties. Please refer to that function for my implementation. ###


def hw2_challenge1c():
    obj_db = np.load('outputs/obj_db.npy')
    img_list = ['many_objects_1.png', 'many_objects_2.png']

    for i in range(len(img_list)):
        labeled_img = Image.open(f'outputs/labeled_{img_list[i]}')
        labeled_img = np.array(labeled_img)
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.

        recognizeObjects(orig_img, labeled_img, obj_db,
                         f'outputs/testing1c_{img_list[i]}')
