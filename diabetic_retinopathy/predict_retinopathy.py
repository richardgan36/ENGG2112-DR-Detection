import numpy as np
from keras import models
from PIL import Image

stage_to_numeric = {"Healthy": 0, "Mild DR": 1, "Moderate DR": 2, "Proliferate DR": 3, "Severe DR": 4}
numeric_to_stage = {v: k for k, v in stage_to_numeric.items()}

model = models.load_model('trained_models/model_3.h5')
test_img = Image.open('/Users/richardgan/Pictures/Machine Learning/diabetic_retinopathy/Severe DR/Severe DR.png')
test_img = test_img.resize((64, 64))
test_img = np.asarray(test_img) / 255
test_img = np.reshape(test_img, (1, 64, 64, 3))

prediction = model.predict(test_img)
print(np.sum(prediction))
print(prediction)
print(numeric_to_stage[np.argmax(prediction)])
