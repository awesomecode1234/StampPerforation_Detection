import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


ideal_width = 53.98
ideal_height = 85.6 

def swap(a, b):
    return b, a
# Helper function to display an image using matplotlib
def show_image(image, title='Image'):
    plt.figure(figsize=(10, 10))
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def getPerforationNum(input_image):
    if isinstance(input_image, str):
        image = cv2.imread(input_image)
    elif isinstance(input_image, np.ndarray):
        image = input_image
    
    # Step 2: Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Step 3: Detect the rotated rectangular area
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    black_contour =contours[0]
    blackrect =cv2.minAreaRect(black_contour)
    blackbox = cv2.boxPoints(blackrect)
    blackbox = np.intp(blackbox)
    largest_contour = contours[1]
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Step 4: Correct the orientation of the stamp
    black_width = int(blackrect[1][0])
    black_height = int(blackrect[1][1])
    stamp_width = int(rect[1][0])
    stamp_height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, stamp_height-1],
                        [0, 0],
                        [stamp_width-1, 0],
                        [stamp_width-1, stamp_height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (stamp_width,stamp_height))
    warped = cv2.resize(warped, (1200, 968))
    # Step 5: Detect Perforations
    width = 1200
    height = 968
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # blurred_warped = cv2.GaussianBlur(gray_warped, (5, 5), 0)

    # edges = cv2.Canny(blurred_warped, 100, 200)


    min_radius = 10
    max_radius = 22
    # Use HoughCircles to detect perforations
    circles = cv2.HoughCircles(gray_warped, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_radius*1.8,
                            param1=100, param2=10, minRadius=min_radius, maxRadius=max_radius)
    top_cnt = 0
    left_cnt = 0
    right_cnt = 0
    bottom_cnt = 0 
    top_y_list = [] 
    bottom_y_list = [] 
    left_x_list = []
    right_x_list = []
    perforation_list = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if not (x >= 0 and x < width and y >=0 and y < height):
                continue        
            if gray_warped[y, x] >= 50:
                continue
            if x < width*0.05:
                flag = False
                for y1 in top_y_list:
                    if abs(y1 - y) < max_radius*2:
                        flag = True
                        break
                if flag:
                    continue
                top_y_list.append(y)
                perforation_list.append((x,y))
                top_cnt += 1  
                cv2.circle(warped, (x, y), r, (0, 255, 0), 2)
                continue
            if y < height*0.05:
                flag = False
                for x1 in right_x_list:
                    if abs(x1 - x) < max_radius*2:
                        flag = True
                        break
                if flag:
                    continue
                right_x_list.append(x)
                perforation_list.append((x,y))
                right_cnt += 1
                cv2.circle(warped, (x, y), r, (0, 255, 0), 2)
                continue
            if x > width*0.95:
                flag = False
                for y1 in bottom_y_list:
                    if abs(y1 - y) < max_radius*2:
                        flag = True
                        break
                if flag:
                    continue
                bottom_y_list.append(y)
                perforation_list.append((x,y))
                bottom_cnt += 1
                cv2.circle(warped, (x, y), r, (0, 255, 0), 2)
                continue
            if y > height*0.95:
                flag = False
                for x1 in left_x_list:
                    if abs(x1 - x) < max_radius*2:
                        flag = True
                        break
                if flag:
                    continue
                left_x_list.append(x)
                perforation_list.append((x,y))
                left_cnt += 1
                cv2.circle(warped, (x, y), r, (0, 255, 0), 2)
                continue
        
        show_image(warped, title='Perforations Detected')
        cv2.imwrite("output.jpg", warped)

        print( stamp_width, stamp_height, black_width, black_height)
        print(f'Top: {top_cnt}, Bottom: {bottom_cnt}, Left: {left_cnt}, Right: {right_cnt}')
        getCompletedList(perforation_list, width, height)
        perforation_width_num = max(top_cnt, bottom_cnt)
        perforation_height_num = max(left_cnt, right_cnt)
        if stamp_width > stamp_height:
            stamp_height, stamp_width = swap(stamp_height, stamp_width)
        stamp_width = stamp_width * (968-20)/968
        stamp_height = stamp_height * (1200-20)/1200        
        if perforation_width_num > perforation_height_num:
           perforation_width_num, perforation_height_num = swap(perforation_width_num, perforation_height_num) 
        if black_width > black_height:
            black_width, black_height = swap(black_width, black_height)

        stamp_width = stamp_width * ideal_width / black_width
        stamp_height = stamp_height * ideal_height / black_height

        num_per_width20 = perforation_width_num * 20 / stamp_width
        num_per_height20 = perforation_height_num * 20  / stamp_height
        return num_per_width20, num_per_height20
    else:
        print("Can not find perforation")
        return None

def modify_list_to_regular_gaps(nums, tolerance=0.3):
    # Calculate the gaps between consecutive numbers
    gaps = [nums[i+1] - nums[i] for i in range(len(nums) - 1)]
    
    # Determine the average gap
    average_gap = sum(gaps) / len(gaps)
    
    # Define a threshold for what we consider similar gaps
    threshold = average_gap * tolerance
    
    # Identify irregular gaps
    irregular_indices = [i for i, gap in enumerate(gaps) if abs(gap - average_gap) > threshold]
    
    modified_list = nums.copy()
    
    # Handle each irregular gap
    for idx in irregular_indices[::-1]:  # reverse to handle indices correctly after modifications
        if gaps[idx] > average_gap + threshold:
            # Calculate how many numbers to add
            n = int(round(gaps[idx] / average_gap))
            for j in range(1, n):
                new_num = nums[idx] + j * average_gap
                modified_list.insert(idx + j, new_num)
        elif gaps[idx] < average_gap - threshold:
            # Remove the smaller gap number
            modified_list.pop(idx + 1)
    
    return modified_list


def getCompletedList(input_list, img_width, img_height):
    
    top_list = sorted([pos[1] for pos in input_list if pos[0] < img_width * 0.1])
    bottom_list = sorted([point[1] for point in input_list if point[0] > img_width * 0.9])
    right_list = sorted([point[0] for point in input_list if point[1] < img_height * 0.1])
    left_list = sorted([point[0] for point in input_list if point[1] > img_height * 0.9])
    return modify_list_to_regular_gaps(top_list), modify_list_to_regular_gaps(bottom_list), modify_list_to_regular_gaps(right_list), modify_list_to_regular_gaps(left_list)  
        


print(getPerforationNum("input_pic/1.jpeg"))
print(getPerforationNum("input_pic/2.jpeg"))
print(getPerforationNum("input_pic/3.jpeg"))

