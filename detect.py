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

def completeList(nums, tolerance=0.2):
    newPosList = []
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
                newPosList.append(new_num)
        # elif gaps[idx] < average_gap - threshold:
        #     # Remove the smaller gap number
        #     modified_list.pop(idx + 1)
    
    return modified_list, newPosList

def custom_round(a):
    # Calculate the remainder when divided by 0.25
    remainder = a % 0.25
    if remainder == 0:
        return a
    # Determine if we need to round up or down
    if remainder < 0.125:
        return a - remainder
    else:
        return a + (0.25 - remainder)

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

    blurred_warped = cv2.GaussianBlur(gray_warped, (5, 5), 0)

    edges = cv2.Canny(blurred_warped, 100, 200)

    min_radius = 9
    max_radius = 22
    # Use HoughCircles to detect perforations
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.3, minDist=min_radius*2,
                            param1=200, param2=0.3, minRadius=min_radius, maxRadius=max_radius)
    radius_cnt = 0
    radius_sum = 0 
    if circles is not None:
        rad_circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in rad_circles:
            if x < max_radius*2.5 or x > width - max_radius*2.5 or y > height - max_radius*2.5 or y< max_radius*2.5:
                radius_sum += r
                radius_cnt += 1

        if radius_cnt == 0:
            return 0,0,0,0
        else:
            radius = int(radius_sum / radius_cnt + 1)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.3, minDist=int(radius*2),
                            param1=200, param2=0.3, minRadius=int(radius*0.3), maxRadius=int(radius*1.5))
        
    top_cnt = 0
    left_cnt = 0
    right_cnt = 0
    bottom_cnt = 0 
    top_y_list = [] 
    bottom_y_list = [] 
    left_x_list = []
    right_x_list = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if not (x >= 0 and x < width and y >=0 and y < height):
                continue        
            if gray_warped[y, x] >= 70:
                continue
            if x < radius*2.5:
                flag = False
                for y1 in top_y_list:
                    if abs(y1 - y) < radius*2:
                        flag = True
                        break
                if flag:
                    continue
                if y > radius*2.7 and y < height - radius*2.7:
                    top_y_list.append(y)
                    cv2.circle(warped, (x, y), r, (0, 255, 0), 2)
            if y < radius*2.5:
                flag = False
                for x1 in right_x_list:
                    if abs(x1 - x) < radius*2:
                        flag = True
                        break
                if flag:
                    continue
                if x > radius*2.7 and x < width - radius*2.7:
                    right_x_list.append(x)
                    cv2.circle(warped, (x, y), r, (0, 255, 0), 2)
            if x > width - radius*2.5:
                flag = False
                for y1 in bottom_y_list:
                    if abs(y1 - y) < radius*2:
                        flag = True
                        break
                if flag:
                    continue
                if y > radius*2.7 and y < height - radius*2.7:
                    bottom_y_list.append(y)
                    cv2.circle(warped, (x, y), r, (0, 255, 0), 2)
            if y > height - radius*2.5:
                flag = False
                for x1 in left_x_list:
                    if abs(x1 - x) < radius*2:
                        flag = True
                        break
                if flag:
                    continue
                if x > radius*2.7 and x < width - radius*2.7:
                    left_x_list.append(x)
                    cv2.circle(warped, (x, y), r, (0, 255, 0), 2)

        modified_List, newPosList = completeList(sorted(top_y_list))        
        for pos in newPosList:
            cv2.circle(warped, (radius, int(pos)), radius, (0, 255),2)
        top_cnt = len(modified_List)

        modified_List, newPosList = completeList(sorted(bottom_y_list))
        for pos in newPosList:
            cv2.circle(warped, (width - radius, int(pos)), radius, (0, 255),2)
        bottom_cnt = len(modified_List)
        
        modified_List, newPosList = completeList(sorted(right_x_list))
        for pos in newPosList:
            cv2.circle(warped, (int(pos), radius), radius, (0, 255), 2)
        right_cnt = len(modified_List)

        modified_List, newPosList = completeList(sorted(left_x_list))
        for pos in newPosList:
            cv2.circle(warped, (int(pos), height - radius), radius, (0, 255),2)
        left_cnt = len(modified_List)

        show_image(warped, title='Perforations Detected')
        cv2.imwrite("output.jpg", warped)

        print(f'Top: {top_cnt}, Bottom: {bottom_cnt}, Left: {left_cnt}, Right: {right_cnt}')
        
        
        perforation_width_num = max (top_cnt, bottom_cnt)
        perforation_height_num = max(left_cnt, right_cnt)
        # if abs(top_cnt-bottom_cnt) > 2:
        #     perforation_width_num = max (top_cnt, bottom_cnt)
        # else:
        #     perforation_width_num = (top_cnt + bottom_cnt)/2
        # if abs(left_cnt-right_cnt) > 2:
        #     perforation_height_num = max (left_cnt, right_cnt)
        # else:
        #     perforation_height_num = (left_cnt + right_cnt)/2
        
        if stamp_width > stamp_height:
            stamp_width, stamp_height = swap(stamp_width, stamp_height)
        stamp_width = stamp_width * (width-radius*3)/width
        stamp_height = stamp_height * (height-radius*2)/height        
        if perforation_width_num > perforation_height_num:
           perforation_width_num, perforation_height_num = swap(perforation_width_num, perforation_height_num) 
        if black_width > black_height:
            black_width, black_height = swap(black_width, black_height)

        stamp_width = stamp_width * ideal_width / black_width
        stamp_height = stamp_height * ideal_height / black_height

        num_per_width20 = perforation_width_num * 20 / stamp_width
        num_per_height20 = perforation_height_num * 20  / stamp_height
        
       
        return custom_round(num_per_width20), custom_round(num_per_height20)
    else:
        # print("Can not find perforation")
        return None


print(getPerforationNum("input_pic/2.jpeg"))
print(getPerforationNum("input_pic/3.jpeg"))
print(getPerforationNum("input_pic/4.jpeg"))
print(getPerforationNum("input_pic/5.jpeg"))
print(getPerforationNum("input_pic/6.jpeg"))
print(getPerforationNum("input_pic/5550.jpeg"))
print(getPerforationNum("input_pic/5553.jpeg"))
