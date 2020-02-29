import cv2
import os
import uuid
from mtcnn.mtcnn import MTCNN

# Метод возвращает область найденного лица изображения
def crop_face(img,detector):
    try:
        faces = detector.detect_faces(img)
        print (faces)
        for face in faces:
            x=face['box'][0]
            y=face['box'][1]
            w=face['box'][2]
            h=face['box'][3]
            img = img[y:y + h, x:x + w]
            return img
    except:
        pass

# Преобразуем набор изображений с kaggle и сохраняем на диск в папку men_new нормализованные лица
if __name__ == '__main__':
    detector = MTCNN()
    folder='men'
    for list_photos in os.walk('dataset_kaggle/{}'.format(folder)):
            for photo in list_photos[2]:
                img = cv2.imread('dataset_kaggle/{}/{}'.format(folder,photo), cv2.COLOR_BGR2GRAY)
                cropped = crop_face(img, detector)

                try:
                    norm_face=cv2.resize(cropped, (200, 200),fx=0.3,fy=0.3, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(r'dataset_kaggle/men_new/{}.jpg'.format(uuid.uuid1()), norm_face)
                except Exception as e:
                    print(e)

