import pathlib

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

img_height = 224
img_width = 224

model = tf.keras.models.load_model('./model_3')

class_names = {'갈비찜': 0, '계란말이': 1, '계란후라이': 2, '김밥': 3, '된장찌개': 4, '떡꼬치': 5, '라면': 6, '물냉면': 7, '미역국': 8, '배추김치': 9, '보쌈': 10, '불고기': 11, '삼겹살': 12, '순대': 13, '유부초밥': 14, '장어구이': 15, '장조림': 16, '족발': 17, '짜장면': 18, '피자': 19, '후라이드치킨': 20}
class_names2 = {v:k for k,v in class_names.items()}

test_img_url = "/home/hoseo/project/test/egg"
img = keras.preprocessing.image.load_img(
    test_img_url, target_size=(img_height, img_width)
)
plt.imshow(img)
plt.show()

img_array = keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, 0) # Create a batch
# 테스트 시각화
prediction_scores = model.predict(img_array)
print(prediction_scores)
predicted_id = np.argmax(prediction_scores)
print(predicted_id)

print(class_names2.get(predicted_id))