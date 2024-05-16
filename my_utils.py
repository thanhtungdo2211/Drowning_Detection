import cv2
import time

def get_fps(img, pTime):
	cTime = time.time()
	fps = 1 / (cTime - pTime)
	pTime = cTime
	cv2.putText(img, f'{int(fps)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
				(255, 0, 0), 5)
	return img, pTime

def resize_img(img, scale):
    w, h,  = int(img.shape[1] * scale / 100), int(img.shape[0] * scale / 100)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img