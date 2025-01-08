This project focuses on grayscale-to-color image conversion using deep learning,
with the objective of restoring historical black-and-white images while evaluating
the structural and color accuracy of the generated outputs. It employs a pretrained
Caffe model, trained on diverse image datasets, to perform automatic colorization.
Additionally, a custom regression model assesses image quality by computing
SSIM (Structural Similarity Index Measure) and PSNR (Peak Signal-to-Noise Ratio) scores.
The structural accuracy achieved by this model is 99%, demonstrating excellent preservation
of image details. However, with PSNR values ranging from 17 to 28 dB, there is room for
improvement, as the ideal PSNR for high-quality colorization is around 30 dB. Since
training the regression model for every new set of images is time-consuming, a PKL (Pickle)
file was created to store the trained model. A PKL file serializes (saves) the model state,
allowing it to be reloaded quickly without retraining, significantly improving efficiency.
A set of 10 image pairs (grayscale and corresponding colorized images) was used for
initial evaluation. To enhance user interaction, there is a simple and attractive interface
built with the Streamlit library that enables users to browse and upload grayscale images for
automatic colorization. A histogram feature further visualizes the frequency of RGB values
along the pixel range, giving insights into the color distribution. This project provides
a holistic approach to automated image restoration, quality analysis, and efficient reuse
of trained models, opening ways for further enhancements.