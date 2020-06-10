import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import logging
import sys

input_directory = "input_directory"
output_directory = "output_directory"

# Some rainbow colours here
pastel = ['#FF6663', '#FEB144', '#FDFD97', '#9EE09E', '#9EC1CF', '#CC99C9']  # standard pastel rainbow
pastel_light = ['#FF9AA2', '#FFB7B2', '#FFDAC1', '#E2F0CB', '#B5EAD7', '#C7CEEA']  # lighter version
pastel2 = ['#ec4e51', '#f59950', '#fce53c', '#7fc04a', '#5f94cb', '#845ea4']
rainbow_seven = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3']


def convert_alpha_to_white(src):
    """
    Convert every pixel with alpha channel into white image.
    The main purpose is to draw hough lines on an image so that you can see clearer.
    """
    image_shape = src.shape
    if image_shape[2] == 3:
        return src

    elif image_shape[2] == 4:
        B, G, R, A = cv2.split(src)
        alpha = A/255
        def convert(alpha, color):
            return (255 * (1 - alpha) + color * alpha).astype(np.uint8)
        B = convert(alpha, B)
        G = convert(alpha, G)
        R = convert(alpha, R)
        src = cv2.merge((B, G, R))
        return src

def convert_src_to_alpha(src):
    """
    Many users do not upload transparent images, which is a huge problem.
    For images without a transparent channel, this function
    converts white pixels into transparent pixels.
    Will not work on all images as a result.
    """
    image_shape = src.shape
    if image_shape[2] == 4:
        return src
    if image_shape[2] == 3:
        # Create alpha shape initiating with opaque values.
        src_alpha = np.full((image_shape[:-1]), 255, dtype = 'uint8')

        # Convert white pixels into transparent pixels.
        alpha_area = np.all(np.absolute([255,255,255] - src[:, :, 0:3]) <= [30,30,30] , axis = -1)
        src_alpha[alpha_area] = 0
        src = np.dstack((src, src_alpha))
        return src

def generate_rainbow_codes(colour_lst , mode = 'bgr'):
    """
    Given a list of hex codes, return a list of colour codes determined from mode.

    generate_rainbow_codes(pastel , mode = 'bgr')
    >> [[99, 102, 255], [68, 177, 254], [151, 253, 253], [158, 224, 158], [207, 193, 158], [201, 153, 204]]
    """
    if mode == 'rgb':
        new_lst = []
        for color in colour_lst:
            new_lst.append(list(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)))
        return new_lst

    if mode == 'bgr':
        new_lst = []
        for color in colour_lst:
            new_lst.append(list(int(color.lstrip('#')[i:i + 2], 16) for i in (4, 2, 0)))
        return new_lst


def detect_optimal_angle(src, export_houghlines = False):
    """
    Detect optimal angles starting from the vertical line (clockwise)
    This means the vertical line has an angle of 0 and a horizontal line has an angle of 150.
    For the adidas logo, the slanted line has an angle of 5pi/6.
    """
    src_white = convert_alpha_to_white(src)
    edges = cv2.Canny(src_white, 50, 150, apertureSize=5)
    threshold_vote = 70 # This is the minimum vote in order to be considered a line. We start with lower limit.

    # We want to generate hough lines but do not want too many. We start with a low threshold number first,
    # if less than 10 lines are detected, we are done
    # if more than 10 lines are detected, we will continue to increase the threshold vote and run houghlines again
    # the loop will stop after running 40 times (picked arbitrary)
    for i in range(40):
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold_vote)
        if lines is None:
            break
        current_line_count = len(lines)
        if current_line_count <= 10:
            break
        elif current_line_count > 10:
            threshold_vote = threshold_vote + 10
    if lines is None:
        return [], src

    # Despite reducing the lines, there are many thetas that are close together.
    # We need to find a way to cluster the 1D data.
    # Find different clusters of thetas using K means
    # Note - Kernel density method probably works as well and both methods require us to vary an input (bandwidth).
    # Iterate through different number of cluster points and find the least number that is able to cluster together well.
    theta_lst = lines[:,:,1]
    for cluster_no in range(1, 6):
        kmeans = KMeans(n_clusters=cluster_no)
        kmeans.fit(theta_lst)
        if kmeans.inertia_ < 0.02 or cluster_no == 5: # cut-off point 5
            theta_array = kmeans.cluster_centers_[:,0]
            unique, counts = np.unique(kmeans.labels_, return_counts = True) # break labels into frequency
            order = counts.argsort() # the order numpy array should be returned
            optimal_angles_lst = theta_array[order[::-1]].tolist() # return optimal angle list in order (most common first)
            break

    # Draw all houghlines for visualization purposes.
    if export_houghlines == True:
        src_debug = src_white.copy()
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 - 3000 * (b))
            y1 = int(y0 + 3000 * (a))
            x2 = int(x0 + 3000 * (b))
            y2 = int(y0 - 3000 * (a))
            cv2.line(src_debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imwrite(f"{output_directory}/houghlines.png", src_debug)
        return optimal_angles_lst, src_debug

    return optimal_angles_lst, src

def find_colour_breakdown(src, colour_space = 'hsv'):
    """
    For images with alpha channel, function extracts all colours and pivots them in a dataframe. Filter away pixels
    with alpha less than 128.
    For images without alpha channel, functions extracts all colours and pivots them in a dataframe.
    We also locate the top left pixel if it's white or black. If Yes, filter them out. This methods sounds very shady
    but it is a quick solution and does work.

    Note that a perceived colour take over a range. For example, white may be [255,255,255] but there may be
    some white pixels that may be [255,254,255]. I am not sure why so I usually specify a range instead.
    """
    if colour_space == 'hsv':
        if src.shape[2] == 4:
            src_alpha = src[:, :, 3]
            src_org = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
            src_hsv = cv2.cvtColor(src_org, cv2.COLOR_BGR2HSV)
            src_hsva = np.dstack((src_hsv, src_alpha))  # concat along third dimension

            pixel_order = src_hsva.reshape(-1, src.shape[-1])  # a numpy array of all pixels
            colors, counter = np.unique(pixel_order, axis=0, return_counts=True)
            colour_df = pd.DataFrame(colors, columns=['h', 's', 'v', 'alpha'])
            colour_df['count'] = counter
            colour_df = colour_df.sort_values(by='count', ascending=False)
            # Exclude white colour (de Morgan's law)
            filter_colour_df = colour_df[(colour_df['h'] != 0) | (colour_df['s'] != 0) | (colour_df['v'] != 255)]
            if filter_colour_df.empty == True:
                filter_colour_df = colour_df
            filter_colour_df = filter_colour_df[filter_colour_df['alpha'] > 128]
            return filter_colour_df.iloc[0][['h', 's', 'v']].tolist()

        elif src.shape[2] == 3:
            if (np.abs(src[0,0] - [0,0,0]) <= np.array([20,20,20])).all():
                background_filter = 0
            elif (np.abs(src[0,0] - [255,255,255]) <= np.array([20,20,20])).all():
                background_filter = 255
            else:
                background_filter = False
            src_org = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
            src_hsv = cv2.cvtColor(src_org, cv2.COLOR_BGR2HSV)

            pixel_order = src_hsv.reshape(-1, src.shape[-1])  # a numpy array of all pixels
            colors, counter = np.unique(pixel_order, axis=0, return_counts=True)
            colour_df = pd.DataFrame(colors, columns=['h', 's', 'v'])
            colour_df['count'] = counter
            colour_df = colour_df.sort_values(by='count', ascending=False)
            if background_filter == 255:
                filter_colour_df = colour_df[(colour_df['v'] <= 240) & (colour_df['s'] >= 20)]
                return filter_colour_df.iloc[0][['h', 's', 'v']].tolist()
            elif background_filter == 0:
                filter_colour_df = colour_df[(colour_df['v'] <= 250)]
                return filter_colour_df.iloc[0][['h', 's', 'v']].tolist()
            else:
                return colour_df.iloc[0][['h', 's', 'v']].tolist()
    else:
        return False

def colour_all(src):
    """
    Returns an array of the image shape (width and height) as True
    """
    return np.full(src[:,:, 0].shape, True)

def colour_common(src):
    """
    Returns an array of the image shape (width and height)
    with the most common colour (obtained from find_colour_breakdown) as True

    We work with the HSV color space because it is a better way to find similar colours through a specified range.
    We take +-20 of the hue from the most common colour obtained colour.
    For the saturation and value, we take a difference of +- 180 and 200 respectively.
    Once again, this value is specified based on experimentation and could probably be optimized.
    Someone mentioned that I can probably use CIELAB and it will probably have better results.
    You can refer to https://en.wikipedia.org/wiki/Color_difference

    Example, if an image has a width of 2 and a height of 2 and the top half of the image is a common colour,
    array shape is:

    array([[ True,  True]   ,
           [ False,  False]])
    """
    src_copy = src.copy()
    src_alpha = src_copy[:,:,3]
    src_alpha_positive = src_alpha >= 50 # to account for weird colours embedded in alpha backgrounds
    src_hsv = cv2.cvtColor(src_copy, cv2.COLOR_BGR2HSV)
    most_common_colour = find_colour_breakdown(src_copy, colour_space='hsv')
    src_hue_positive = np.all(np.remainder(src_hsv[:,:,0:1] - most_common_colour[0:1], 180) <= 20, axis=-1)
    src_hue_negative = np.all(np.remainder(most_common_colour[0:1] - src_hsv[:,:,0:1], 180) <= 20, axis=-1)
    src_hue = np.logical_or(src_hue_positive, src_hue_negative)

    src_satval = np.all(np.abs(src_hsv[:,:,1:] - most_common_colour[1:]) <= [180,200], axis=-1)

    src_coloured = np.logical_and(src_hue, src_satval)
    src_coloured = np.logical_and(src_coloured, src_alpha_positive)
    return src_coloured

def colour_blackwhite(src, reverse = True):
    """
    Returns an array of the image shape (width and height) with non-black and white pixels as True.
    For just black and white pixels, change reverse to False.

    Not necessary to convert this into HSV colour space.

    I didn't experiment with this function yet.
    """
    src_copy = src.copy()
    src_black = np.all(src_copy[:, :, 0:3] - [0,0,0] <= [10,10,10], axis = -1)
    src_white = np.all([255,255,255] - src_copy[:, :, 0:3] <= [20,20,20], axis = -1)
    src_combined = np.logical_or(src_white, src_black)
    if reverse == True:
        return ~src_combined
    else:
        return src_combined

def transform_axis(src, colour_method, colour_lst, axis = 'horizontal'):
    """
    Transform an alpha source with a colour method by axis.

    colour_method can either be colour_all, colour_common or colour_blackwhite.
    axis can either be horizontal or vertical. For angles, refer to function transform_optimal
    """
    src_copy = src.copy()
    # Identify colour region
    src_coloured = colour_method(src_copy)

    # Identify target area to change (left and right limited of coloured region)
    y, x = src_coloured.nonzero() # return all the y and x pairings with non-zero coordinates
    top_row, btm_row = y.min(), y.max()
    left_row, right_row = x.min(), x.max()
    src_target = src_copy[top_row:btm_row, left_row:right_row, :]

    # Initiate the target region (for the respective colours in the rainbow) to be False.
    # We will change the target region to True by looping them.
    target_region = np.full(src_target[:, :, 0].shape, False)

    # To align the shape with target_region
    coloured_region = src_coloured[top_row:btm_row, left_row:right_row]

    colour_lst = generate_rainbow_codes(colour_lst, 'bgr')
    for counter in range(len(colour_lst)):
        striped_region = np.copy(target_region)
        # Apply broadcasting of rainbow colours to targeted region of image (without changing alpha channel)
        if axis == 'vertical':
            thickness = (right_row - left_row) / len(colour_lst)
            striped_region[:, int(thickness*counter): int(thickness * (counter + 1))] = True
        elif axis == 'horizontal':
            thickness = (btm_row - top_row) / len(colour_lst)
            striped_region[int(thickness * counter): int(thickness * (counter + 1)), :] = True
        else:
            return False
        # Takes the AND condition of both striped region (the fraction of pixels for the rainbow component) and
        # coloured_region (pixels that are supposed to be coloured)
        # For visualization, refer to Figure 2.1 in the README file.
        combined_region = np.logical_and(striped_region, coloured_region)

        # Apply the colour to affected region
        src_target[:,:,0:3][combined_region] = colour_lst[counter]

    # Since src_target is a subset of the actual image, we replaced the original image with src_target
    src_output = np.copy(src)
    src_output[top_row:btm_row, left_row:right_row, :] = src_target
    return src_output

def transform_optimal(src, colour_method, colour_lst):
    """
    Transform an alpha source with a colour method by axis.

    colour_method can either be colour_all, colour_common or colour_blackwhite.
    There is no axis here as we are aiming to find the best angles. Horizontal (pi/2) and Vertical (0) are ignored.
    """
    src_lst, src_ort = [], []
    src_copy = src.copy()
    src_coloured = colour_method(src_copy)

    # Finding the left and right limit of the actual logo is non-trivial here (Refer to Figure 2.2 in README.md)
    # For the horizontal case, I can just get the x.min() and x.max() for the left and right limit respectively
    # However, for angles, we need to find the starting point.
    # I find that using the Pythagoras' theorem works best (Refer to Figure 2.2 in README.md) but there are some
    # limitations. To counter the limitations, I covered the remaining areas with colour just in case.
    y_lst, x_lst = src_coloured.nonzero()  # return all the y and x pairings with non-zero coordinates
    y_global_max, x_global_max = src_coloured.shape
    # I have to calculate the Pythagoras distance for every single corner.
    # This is memory intensive but it's pretty fast due to vectorization (from numpy).
    pythagoras_btmleft = (x_lst) ** 2 + (y_lst - y_global_max) ** 2
    pythagoras_topright = (x_lst - x_global_max) ** 2 + (y_lst) ** 2
    pythagoras_origin = x_lst ** 2 + y_lst ** 2
    pythagoras_btmright = (x_lst - x_global_max) ** 2 + (y_lst - y_global_max) ** 2

    # Find the pixel that is closest to each corner.
    # For example, index_btmleft is the x and y coordinate from the btm left corner of image.
    index_btmleft = np.where(pythagoras_btmleft == pythagoras_btmleft.min())
    index_topright = np.where(pythagoras_topright == pythagoras_topright.min())
    index_origin = np.where(pythagoras_origin == pythagoras_origin.min())
    index_btmright = np.where(pythagoras_btmright == pythagoras_btmright.min())

    optimal_angle_lst = detect_optimal_angle(src, export_houghlines=True)[0]
    colour_lst = generate_rainbow_codes(colour_lst, 'bgr')

    for optimal_angle in optimal_angle_lst[:3]:
        # Ignore all vertical/horizontal (Deal separately with transform_axis)
        if abs(optimal_angle - 0) < 0.05 or abs(optimal_angle - np.pi/2) < 0.05 or abs(optimal_angle - np.pi ) < 0.05:
            continue

        # Case 1 - if theta falls between 90 (pi/2) to 180 (pi) [third quadrant]
        src_target = src.copy()
        target_region = np.full(src_target[:, :, 0].shape, False)
        coloured_region = src_coloured.copy()
        if np.pi/2 < optimal_angle < np.pi:
            theta = optimal_angle - np.pi/2
        else:
            theta = np.pi/2 - optimal_angle

        # Determine all the equations. Refer to Figure 2.2 in the README.md file.
        gradient = - np.tan(theta)
        top_equation_y_intercept = - (y_lst[index_topright[0][0]]) - (gradient * (x_lst[index_topright[0][0]]))
        btm_equation_y_intercept = - (y_lst[index_btmleft[0][0]]  ) - (gradient * (x_lst[index_btmleft[0][0]]))
        y_intercept_gap = (top_equation_y_intercept - btm_equation_y_intercept) / len(colour_lst)

        for counter in range(len(colour_lst)):
            striped_region = np.copy(target_region)
            previous_y_intercept = btm_equation_y_intercept + (counter) * y_intercept_gap
            current_y_intercept = btm_equation_y_intercept + (counter+1) * y_intercept_gap
            # Now the below method is really inefficient and look very naive.
            # I basically looped through every pixel but to justify myself,
            # this method is really readable and I'm running tight on the time I'm allowed to spend on this project =p
            for y, row in enumerate(striped_region):
                for x, target in enumerate(row):
                    # the reason y is negative because we are treating top left as origin (y is moving downwards)
                    if ((gradient * x) + previous_y_intercept) < - y and - y < (gradient * x) + current_y_intercept:
                        striped_region[y][x] = True
            # Similar to above (transform_axis), takes the AND condition of the two regions.
            combined = np.logical_and(striped_region, coloured_region)
            src_target[:,:,0:3][combined] = colour_lst[counter]

        # I explained above the Pythagoras method has some limitations. Some end edges may not be covered.
        # This serves to cover the edges. Refer to Figure 2.2 (area below red line and area above gray diagonal line)
        striped_region = np.copy(target_region)
        extreme_left_y_intercept =  btm_equation_y_intercept + (- 2) * y_intercept_gap
        left_y_intercept =  btm_equation_y_intercept + (0) * y_intercept_gap
        for y, row in enumerate(striped_region):
            for x, target in enumerate(row):
                # the reason y is negative because we are treating top left as origin (y is moving downwards)
                if ((gradient * x) + extreme_left_y_intercept) < - y and - y < (gradient * x) + left_y_intercept:
                    striped_region[y][x] = True
        combined = np.logical_and(striped_region, coloured_region)
        src_target[:, :, 0:3][combined] = colour_lst[0]

        striped_region = np.copy(target_region)
        extreme_right_y_intercept = btm_equation_y_intercept + (len(colour_lst) + 1) * y_intercept_gap
        right_y_intercept = btm_equation_y_intercept + (len(colour_lst) - 1) * y_intercept_gap
        for y, row in enumerate(striped_region):
            for x, target in enumerate(row):
                # the reason y is negative because we are treating top left as origin (y is moving downwards)
                if ((gradient * x) + right_y_intercept) < - y and - y < (gradient * x) + extreme_right_y_intercept:
                    striped_region[y][x] = True
        combined = np.logical_and(striped_region, coloured_region)
        src_target[:, :, 0:3][combined] = colour_lst[-1]

        src_output = np.copy(src)
        src_output = src_target
        src_lst.append(src_output)

        # Case 2 - if theta falls between 0 to 90 (pi/2) [first quadrant]. This is orthogonal to Case 1.
        # The below code is exactly the same as the top one so I'm not leaving any comments here.
        src_target = src.copy()
        target_region = np.full(src_target[:, :, 0].shape, False)
        coloured_region = src_coloured.copy()
        theta_perpendicular = np.pi/2 - theta
        gradient = np.tan(theta_perpendicular)
        top_equation_y_intercept = - (y_lst[index_origin[0][0]] ) - (gradient * (x_lst[index_origin[0][0]] ) )
        btm_equation_y_intercept = - (y_lst[index_btmright[0][0]] ) - (gradient * (x_lst[index_btmright[0][0]] ) )
        y_intercept_gap = (top_equation_y_intercept - btm_equation_y_intercept) / len(colour_lst)

        for counter in range(len(colour_lst)):
            striped_region = np.copy(target_region)
            previous_y_intercept = top_equation_y_intercept - (counter) * y_intercept_gap
            current_y_intercept = top_equation_y_intercept - (counter + 1) * y_intercept_gap
            for y, row in enumerate(target_region):
                for x, target in enumerate(row):
                    if ((gradient * x) + previous_y_intercept) > - y and - y > (gradient * x) + current_y_intercept:
                        striped_region[y][x] = True
            combined_region = np.logical_and(striped_region, coloured_region)
            src_target[:, :, 0:3][combined_region] = colour_lst[counter]

        # Cover edges
        striped_region = np.copy(target_region)
        extreme_left_y_intercept = top_equation_y_intercept - (- 2) * y_intercept_gap
        left_y_intercept = top_equation_y_intercept - (0) * y_intercept_gap
        for y, row in enumerate(striped_region):
            for x, target in enumerate(row):
                if ((gradient * x) + extreme_left_y_intercept) > - y and - y > (gradient * x) + left_y_intercept:
                    striped_region[y][x] = True
        combined = np.logical_and(striped_region, coloured_region)
        src_target[:, :, 0:3][combined] = colour_lst[0]

        striped_region = np.copy(target_region)
        extreme_right_y_intercept = top_equation_y_intercept - (len(colour_lst) - 1) * y_intercept_gap
        right_y_intercept = top_equation_y_intercept - (len(colour_lst) + 1) * y_intercept_gap
        for y, row in enumerate(striped_region):
            for x, target in enumerate(row):
                if ((gradient * x) + extreme_right_y_intercept) > - y and - y > (gradient * x) + right_y_intercept:
                    striped_region[y][x] = True
        combined = np.logical_and(striped_region, coloured_region)
        src_target[:, :, 0:3][combined] = colour_lst[-1]

        src_output = np.copy(src)
        src_output = src_target
        src_ort.append(src_output)

    return src_lst, src_ort

def generate_logo_wrapper(logo):
    """
    Wrapper function to generate images for various methods.
    """
    try:
        compile_lst = []
        src = cv2.imread(f"input_directory/{logo}", cv2.IMREAD_UNCHANGED)
        if src is None:
            raise Exception("Unable to open image, please insert your image into [input_directory]. Example [python rainbow-logos.py yourimage.png]")
        file_name, file_format = logo.split(".")
        height, width, channels = src.shape
        if channels == 3:
            logging.warning("Your photo doesn't have the alpha channel. Please use .png or transparent photos for better results!")
        src = convert_src_to_alpha(src)


        src_lst_all, src_ort_all = transform_optimal(src, colour_all, pastel)
        src_lst_common, src_ort_common = transform_optimal(src, colour_common, pastel)
        src_hort_all = transform_axis(src, colour_all, pastel, axis='horizontal')
        src_hort_common = transform_axis(src, colour_common, pastel, axis='horizontal')
        src_vert_all = transform_axis(src, colour_all, pastel, axis='vertical')
        src_vert_common = transform_axis(src, colour_common, pastel, axis='vertical')

        file_format = 'png'
        if len(src_lst_all) != 0:
            cv2.imwrite(f"{output_directory}/{file_name}_optimalparallel_all.{file_format}", src_lst_all[0],
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
        if len(src_ort_all) != 0:
            cv2.imwrite(f"{output_directory}/{file_name}_optimalperpendicular_all.{file_format}", src_ort_all[0],
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
        if len(src_lst_common) != 0:
            cv2.imwrite(f"{output_directory}/{file_name}_optimalparallel_common.{file_format}", src_lst_common[0],
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
        if len(src_ort_common) != 0:
            cv2.imwrite(f"{output_directory}/{file_name}_optimalperpendicular_common.{file_format}",
                        src_ort_common[0],
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(f"{output_directory}/{file_name}_hort_all.{file_format}", src_hort_all,
                    [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(f"{output_directory}/{file_name}_hort_common.{file_format}", src_hort_common,
                    [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(f"{output_directory}/{file_name}_vert_all.{file_format}", src_vert_all,
                    [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(f"{output_directory}/{file_name}_vert_common.{file_format}", src_vert_common,
                    [cv2.IMWRITE_JPEG_QUALITY, 100])
        logging.warning("Process completed! Open the output_directory folder for your results!")
        return True

    except Exception as e:
        logging.critical(f"{e}")
        return False

if __name__ == "__main__":
    logo = sys.argv[1]
    generate_logo_wrapper(logo)

"""
# For testing in your IDE
# Insert files into the input_directory

logo = "adidas.png"
generate_logo_wrapper(logo)
"""
