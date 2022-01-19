import paddlehub as hub
import cv2

img_path = "2.jpeg"
img=cv2.imread(img_path)

model=hub.Module(name='detect_test')
print(model.predict(img))