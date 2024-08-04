import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


ideal_width = 53.98
ideal_height = 85.6 
width = 1200
height = 968

def swap(a, b):
    return b, a
# Helper function to display an image using matplotlib
# def show_image(image, title='Image'):
#     plt.figure(figsize=(10, 10))
#     if len(image.shape) == 3:
#         plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     else:
#         plt.imshow(image, cmap='gray')
#     plt.title(title)
#     plt.axis('off')
#     plt.show()

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

def findPerforationPerLength(perfor_list, pixel_per_1cm):
    cntList = []
    for i in np.arange(perfor_list[0], perfor_list[-1]+ 0.1 - pixel_per_1cm, 0.1):
        cnt = 0
        for j in perfor_list:
            if i<= j <= i+pixel_per_1cm:
                cnt += 1
        cntList.append(cnt)
    if len(cntList) > 2:
        greatest = max(cntList)
        cntList.remove(greatest)
        smallest = min(cntList)
        cntList.remove(smallest)

    average = sum(cntList)/len(cntList)
    return average
    

def getPerforationNum(input_image):
    if isinstance(input_image, str):
        image = cv2.imread(input_image)
    elif isinstance(input_image, np.ndarray):
        image = input_image
    
    # Step 2: Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # denoised_image = cv2.GaussianBlur(gray, (5,5), 0)
    denoised_image = cv2.medianBlur(gray, 5)
    
    denoised_image = cv2.fastNlMeansDenoising(denoised_image, None, 100, 10, 21)
    _, binary_image = cv2.threshold(denoised_image, 70, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # Apply opening (erosion followed by dilation)
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    # Step 3: Detect the rotated rectangular area
    contours, _ = cv2.findContours(opened_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    black_contour =contours[0]
    blackrect =cv2.minAreaRect(black_contour)
    blackbox = cv2.boxPoints(blackrect)
    blackbox = np.intp(blackbox)
    stamp_contour = contours[1]
    stamp_rect = cv2.minAreaRect(stamp_contour)
    box = cv2.boxPoints(stamp_rect)
    stamp_box = np.intp(box)
    
    # Step 4: Correct the orientation of the stamp
    black_width = int(blackrect[1][0])
    black_height = int(blackrect[1][1])
    if black_width > black_height:
        black_width, black_height = swap(black_width, black_height)
    stamp_width = int(stamp_rect[1][0])
    stamp_height = int(stamp_rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, stamp_height-1],
                        [0, 0],
                        [stamp_width-1, 0],
                        [stamp_width-1, stamp_height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    gray_warped = cv2.warpPerspective(opened_image, M, (stamp_width,stamp_height))
    if stamp_width > stamp_height:
        gray_warped = cv2.rotate(gray_warped, cv2.ROTATE_90_CLOCKWISE)
        stamp_width, stamp_height = swap(stamp_width, stamp_height)
    # print(black_height, black_width, stamp_height, stamp_width)
    
    stamp_img = cv2.resize(gray_warped, (stamp_width, stamp_height))
    # warped = cv2.resize(warped, (1200, 968))
    # Step 5: Detect Perforations
    edges = cv2.Canny(stamp_img, 100, 200)
    
    min_radius = 8
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
    else:
        return 0,0,0,0
    
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.3, minDist=int(radius*1.9),
                            param1=200, param2=0.3, minRadius=int(radius*0.3), maxRadius=int(radius*1.5))
        
    top_cnt = 0
    left_cnt = 0
    right_cnt = 0
    bottom_cnt = 0 
    top_x_list = [] 
    bottom_x_list = [] 
    left_y_list = []
    right_y_list = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:

            if not (x >= 0 and x < stamp_width and y >=0 and y < stamp_height):
                continue        
            if gray_warped[y, x] <= 70:
                continue

            if y < radius*2.5:
                flag = False
                for x1 in top_x_list:
                    if abs(x1 - x) < radius*2:
                        flag = True
                        break
                if flag:
                    continue
                if x > radius*2.7 and x < stamp_width - radius*2.7:
                    top_x_list.append(x)
                    cv2.circle(stamp_img, (x, y), r, (0, 255, 0), 2)

            if x < radius*2.5:
                flag = False
                for y1 in left_y_list:
                    if abs(y1 - y) < radius*1.9:
                        flag = True
                        break
                if flag:
                    continue
                if y > radius*2.7 and y < stamp_height - radius*2.7:
                    left_y_list.append(y)
                    cv2.circle(stamp_img, (x, y), r, (0, 255, 0), 2)

            if y > stamp_height - radius*2.5:
                flag = False
                for x1 in bottom_x_list:
                    if abs(x1 - x) < radius*2:
                        flag = True
                        break
                if flag:
                    continue
                if x > radius*2.7 and x < stamp_width - radius*2.7:
                    bottom_x_list.append(x)
                    cv2.circle(stamp_img, (x, y), r, (0, 255, 0), 2)

            if x > stamp_width - radius*2.5:
                flag = False
                for y1 in right_y_list:
                    if abs(y1 - y) < radius*2:
                        flag = True
                        break
                if flag:
                    continue
                if y > radius*2.7 and y < stamp_height - radius*2.7:
                    right_y_list.append(y)
                    cv2.circle(stamp_img, (x, y), r, (0, 255, 0), 2)

        
        modified_List, newPosList = completeList(sorted(top_x_list))        
        for pos in newPosList:
            cv2.circle(stamp_img, (radius, int(pos)), radius, (0, 255),2)
        top_cnt = len(modified_List)
        top_x_list = sorted(modified_List)

        modified_List, newPosList = completeList(sorted(bottom_x_list))
        for pos in newPosList:
            cv2.circle(stamp_img, (width - radius, int(pos)), radius, (0, 255),2)
        bottom_cnt = len(modified_List)
        bottom_x_list = sorted(modified_List)

        modified_List, newPosList = completeList(sorted(right_y_list))
        for pos in newPosList:
            cv2.circle(stamp_img, (int(pos), radius), radius, (0, 255), 2)
        right_cnt = len(modified_List)
        right_y_list = sorted(modified_List)

        modified_List, newPosList = completeList(sorted(left_y_list))
        for pos in newPosList:
            cv2.circle(stamp_img, (int(pos), height - radius), radius, (0, 255),2)
        left_cnt = len(modified_List)
        left_y_list = sorted(modified_List)

        # show_image(stamp_img, title='Perforations Detected')

        # print(f'Top: {top_cnt}, Bottom: {bottom_cnt}, Left: {left_cnt}, Right: {right_cnt}')
        
        # perforation_width_num = max (top_cnt, bottom_cnt)
        # perforation_height_num = max(left_cnt, right_cnt)
        # if abs(top_cnt-bottom_cnt) > 2:
        #     perforation_width_num = max (top_cnt, bottom_cnt)
        # else:
        #     perforation_width_num = (top_cnt + bottom_cnt)/2
        # if abs(left_cnt-right_cnt) > 2:
        #     perforation_height_num = max (left_cnt, right_cnt)
        # else:
        #     perforation_height_num = (left_cnt + right_cnt)/2
        pixel_per_mm = black_width / ideal_width
        pixel_per_2cm = pixel_per_mm * 10
        
        # print(top_x_list, bottom_x_list, left_y_list, right_y_list)
        num_per_top = findPerforationPerLength(top_x_list, pixel_per_2cm) 
        num_per_bottom =findPerforationPerLength(bottom_x_list, pixel_per_2cm)
        num_per_right = findPerforationPerLength(right_y_list, pixel_per_2cm)
        num_per_left = findPerforationPerLength(left_y_list, pixel_per_2cm)
        
        

        # stamp_width = stamp_width * (width-radius*2)/width
        # stamp_height = stamp_height * (height-radius*2)/height        

        # stamp_width = stamp_width * ideal_width / black_width
        # stamp_height = stamp_height * ideal_height / black_height

        # num_per_width20 = perforation_width_num * 20 / stamp_width
        # num_per_height20 = perforation_height_num * 20  / stamp_height
       
        return custom_round(min(num_per_top, num_per_bottom)*2), custom_round(min(num_per_top, num_per_bottom)*2), custom_round(min(num_per_left, num_per_right)*2),custom_round(min(num_per_left, num_per_right)*2) 
    else:
        # print("Can not find perforation")
        return None


# print(getPerforationNum("input_pic/4.jpeg"))
