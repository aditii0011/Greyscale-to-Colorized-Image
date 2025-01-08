import numpy as np
import cv2
import streamlit as st
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from cv2 import dnn
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="BW to Colorized Image",
    page_icon="🎨",
    layout="centered"
)

proto_file = r'colorization_deploy_v2.prototxt'
model_file = r'colorization_release_v2.caffemodel'
hull_pts = r'pts_in_hull.npy'

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
    st.pyplot(plt)

st.title("BW to Colorized Image")
uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "png"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    grayscale_path = "temp.jpg"
    
    colorized_img = colorize_image(grayscale_path)
    
    grayscale_img = cv2.imread(grayscale_path)
    grayscale_img = cv2.cvtColor(grayscale_img, cv2.COLOR_BGR2RGB)
    colorized_img_rgb = cv2.cvtColor(colorized_img, cv2.COLOR_BGR2RGB)
    
    st.subheader("Grayscale and Colorized Images")
    col1, col2 = st.columns([2, 2])
    with col1:
        st.image(grayscale_img, caption="Grayscale Image", use_container_width=True)
    with col2:
        st.image(colorized_img_rgb, caption="Colorized Image", use_container_width=True)
    
    st.subheader("RGB Histogram of the Colorized Image")
    plot_rgb_histogram(colorized_img)
    
    original_color = cv2.imread(grayscale_path.replace("gray", "color"))
    ssim_value, _ = compare_ssim(cv2.cvtColor(original_color, cv2.COLOR_BGR2GRAY),
                                 cv2.cvtColor(colorized_img, cv2.COLOR_BGR2GRAY), full=True)
    psnr_value = compare_psnr(original_color, colorized_img)
    print(f"SSIM: {ssim_value}")
    print(f"PSNR: {psnr_value}")