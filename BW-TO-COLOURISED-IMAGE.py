import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from cv2 import dnn
import joblib
import os
import matplotlib.pyplot as plt

proto_file = r'colorization_deploy_v2.prototxt'
model_file = r'colorization_release_v2.caffemodel'
hull_pts = r'pts_in_hull.npy'
training_data = [
    (r'images\gray1.jpg', r'images\color1.jpg'),
    (r'images\gray2.jpg', r'images\color2.jpg'),
    (r'images\gray3.jpg', r'images\color3.jpg'),
    (r'images\gray4.jpg', r'images\color4.jpg'),
    (r'images\gray5.jpg', r'images\color5.jpg'),
    (r'images\gray6.jpg', r'images\color6.jpg'),
    (r'images\gray7.jpg', r'images\color7.jpg'),
    (r'images\gray8.jpg', r'images\color8.jpg'),
    (r'images\gray9.jpg', r'images\color9.jpg'),
    (r'images\gray10.jpg', r'images\color10.jpg'),
]

net = dnn.readNetFromCaffe(proto_file, model_file)
kernel = np.load(hull_pts)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(grayscale_image_path):

    grayscale = cv2.imread(grayscale_image_path, cv2.IMREAD_GRAYSCALE)
    grayscale = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

    img = grayscale.astype("float32") / 255.0
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    height, width = lab_img.shape[:2]
    L = cv2.split(lab_img)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(cv2.resize(L, (224, 224))))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (width, height))

    L = cv2.split(lab_img)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return colorized

def resize_image(image, max_width, max_height):
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    return cv2.resize(image, new_size)

def extract_features_and_metrics(grayscale_path, color_path):
    colorized = colorize_image(grayscale_path)
    original_color = cv2.imread(color_path)

    ssim_value, _ = compare_ssim(cv2.cvtColor(original_color, cv2.COLOR_BGR2GRAY),
                                 cv2.cvtColor(colorized, cv2.COLOR_BGR2GRAY), full=True)
    psnr_value = compare_psnr(original_color, colorized)

    colorized_hist = cv2.calcHist([colorized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    features = colorized_hist
    metrics = [ssim_value, psnr_value]

    return features, metrics

def plot_rgb_histogram(image):
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(10, 5))
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.title('RGB Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

model_file_path = 'regression_model.pkl'
if not os.path.exists(model_file_path):
    # Training phase
    X_train = []
    y_train = []

    for grayscale_path, color_path in training_data:
        features, metrics = extract_features_and_metrics(grayscale_path, color_path)
        X_train.append(features)
        y_train.append(metrics)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train a regression model (e.g., Linear Regression)
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    joblib.dump(reg_model, model_file_path)
    print("Model training completed and saved to 'regression_model.pkl'")
else:
    print("Model already exists. Skipping training.")

# Predicting phase
def load_model_and_predict_accuracy(grayscale_path):
    
    reg_model = joblib.load(model_file_path)

    colorized = colorize_image(grayscale_path)
    features = cv2.calcHist([colorized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    predicted_metrics = reg_model.predict([features])
    ssim_pred, psnr_pred = predicted_metrics[0]

    print(f"Predicted SSIM: {ssim_pred}")
    print(f"Predicted PSNR: {psnr_pred}")

    grayscale = cv2.imread(grayscale_path, cv2.IMREAD_GRAYSCALE)
    grayscale = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

    max_display_width, max_display_height = 640, 480
    grayscale_resized = resize_image(grayscale, max_display_width, max_display_height)
    colorized_resized = resize_image(colorized, max_display_width, max_display_height)

    result = cv2.hconcat([grayscale_resized, colorized_resized])
    cv2.imshow("Grayscale -> Colorized", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plot_rgb_histogram(colorized)

new_grayscale_path = r'greyscale-closeup-shot-of-an-angry-wolf-with-a-blurred-background-ai-generated-free-photo.jpg'
load_model_and_predict_accuracy(new_grayscale_path)