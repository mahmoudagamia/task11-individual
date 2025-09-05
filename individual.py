import cv2
import numpy as np

def detect_shapes_and_colors(image_path):
    # Load the image and resize it for consistent processing
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    img = cv2.resize(img, (800, 600))
    
    # Create a copy to draw the final results on
    output_image = img.copy()

    # Convert the image from BGR to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV color ranges for each color
    # Note: Red wraps around the hue scale, so it needs two ranges.
    color_ranges = {
        'Red': ([0, 120, 70], [10, 255, 255], [170, 120, 70], [180, 255, 255]),
        'Green': ([35, 100, 50], [85, 255, 255], None, None),
        'Blue': ([90, 100, 50], [130, 255, 255], None, None),
        'Yellow': ([20, 100, 100], [30, 255, 255], None, None),
    }

    # Iterate over each color and its defined range
    for color_name, ranges in color_ranges.items():
        # Create the mask(s) for the current color
        lower1 = np.array(ranges[0])
        upper1 = np.array(ranges[1])
        mask1 = cv2.inRange(hsv_img, lower1, upper1)

        mask = mask1
        # dont know what that shit is honestly 
        if ranges[2] is not None:
            lower2 = np.array(ranges[2])
            upper2 = np.array(ranges[3])
            mask2 = cv2.inRange(hsv_img, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        
        # Find contours on the mask for the current color
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour found
        for contour in contours:
            # ignore noise i think
            if cv2.contourArea(contour) < 200:
                continue

            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Determine the shape name based on the number of vertices
            sides = len(approx)
            shape_name = ''
            if sides == 3:
                shape_name = 'Triangle'
            elif sides == 4:
                # Differentiate between square and rectangle
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.95 <= aspect_ratio <= 1.05:
                    shape_name = 'Square'
                else:
                    shape_name = 'Rectangle'
            else:
               
                shape_name = 'Circle'

            # get the cente
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 0, 0
            
            # Create the final label with color and shape
            label = f"{color_name} {shape_name}"
            
            # Draw the contour and the label on the output image
            cv2.drawContours(output_image, [approx], -1, (0, 0, 0), 3) # Black outline
            cv2.putText(output_image, label, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Display shape an col
    cv2.imshow('Detected Shapes and Colors', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_file_path = '/home/agamia/Downloads/test.jpg'
    detect_shapes_and_colors(image_file_path)