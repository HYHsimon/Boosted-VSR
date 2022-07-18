import cv2
import numpy as np

def dis_flow(img0, img1):

    img0 = np.asanyarray(img0.permute(1, 2, 0).cpu() * 255, dtype="uint8")
    img1 = np.asanyarray(img1.permute(1, 2, 0).cpu() * 255, dtype="uint8")

    dis = cv2.DISOpticalFlow_create(2)

    img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    dis_flow = dis.calc(img0_gray, img1_gray, None, )

    return np.mean(abs(dis_flow))