from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import cv2

image_path = "path/to/your/image.jpg"
image = cv2.imread(image_path)

blue_channel, green_channel, red_channel = cv2.split(image)

gray_image = cv2.cvtColor(image, cv2.COLORBGR2GRAY)
, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)


X = binary_image.reshape(-1, 1)
y = np.array([1 if pixel == 255 else 0 for pixel in X])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


classifier = GaussianNB()
classifier.fit(X_train, y_train)

Hace predicciones en el conjunto de prueba
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
