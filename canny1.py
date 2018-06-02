import cv2
import numpy as np

def segment (image):
    image_gray = cv2.cvtColor(image, cv2.BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)

    edges = cv2.Canny(image_gray, 20, 60)
    countours = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(edges, contours, -1, 255, 2)
    inv_edges = cv2.bitwise_not(edges)
    retval, rect = cv2.floodFill(inv_edges, None, (0, 0), 0)
    mask = cv2.bitwise_or(edges, inv_edges)

    result = np.zeros(image,shape, dtype = 'uint8')
    result[mask > 0, :] = image[mask > 0, :]

    return  result

if __name__ == '__main__':
    image = cv2.imread('foto.jpg')
    result = segment(image)
    cv2.imwrite('result.jpg', result)