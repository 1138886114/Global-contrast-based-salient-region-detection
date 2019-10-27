import numpy as np
import time
import cv2



def LC(image_gray):
    image_height,image_width = image_gray.shape[:2]
    base_matrix = np.zeros((256,256),dtype=np.int32)
    base_line = np.array(range(256))

    base_matrix[0] = base_line
    for i in range(1,256):
        base_matrix[i] = np.roll(base_line,i)
    base_matrix = np.triu(base_matrix)
    temp_matrix = np.triu(base_matrix).T
    base_matrix = np.add(base_matrix ,temp_matrix)

    hist_array = cv2.calcHist([image_gray], [0], None, [256], [0.0, 256.0])
    hist_reshape = hist_array.reshape(1,256)
    temp_matrix = np.tile(hist_reshape, (256, 1))
    sum_diag_hist = np.sum(np.multiply(base_matrix,temp_matrix),axis=1)

    image_gray_value = image_gray.reshape(1,image_height*image_width)[0]
    image_gray_copy = np.zeros(image_height * image_width, dtype=np.int32)

    for i,x in enumerate(image_gray_value):
        image_gray_copy[i] = sum_diag_hist[x]
    image_result = image_gray_copy.reshape(image_height,image_width)
    image_result = (image_result-np.min(image_result))/(np.max(image_result)-np.min(image_result))
    return image_result


if __name__ == '__main__':
    file = r"C:\Users\xxx\Desktop\001.png"

    start = time.time()
    image_gray = cv2.imread(file, 0)
    saliency_image = LC(image_gray)
    # cv2.imwrite(r"C:\Users\xxx\Desktop\001_1.png",saliency_image*255)
    end = time.time()

    print("Duration: %.2f seconds." % (end - start))
    cv2.imshow("gray saliency image", saliency_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()