# Neural Style Transfer (NST) with PyTorch

## Overview
This project implements **Neural Style Transfer (NST)**, a computer vision technique that blends the *content* of one image with the *artistic style* of another. Using deep learning, the system can render a standard photograph in the style of famous artworks (e.g., Van Gogh's *Starry Night*).

This implementation focuses on leveraging pre-trained Convolutional Neural Networks (CNNs) to manipulate image feature representations, demonstrating a strong understanding of feature extraction and loss optimization in generative AI.

## Technical Implementation
The core of this project is built on **PyTorch** and utilizes the **VGG19** architecture for feature extraction.

**Key Features:**
* **VGG19 Backbone:** Utilized the `torchvision.models` VGG19 network (pretrained on ImageNet) to extract high-level content features and low-level style textures.
* **Custom Loss Function:** Implemented a composite loss function combining:
    * **Content Loss:** Ensures the spatial structure of the original photo remains intact.
    * **Style Loss:** Matches the Gram matrices of feature maps to capture artistic textures.
* **Gradient Descent Optimization:** Instead of training a network weights, the optimization is performed directly on the *input image pixel values* to minimize the total loss.
* **CPU Optimization:** Engineered the pipeline to be lightweight and efficient, enabling execution on CPU-only environments (including standard MacOS) without requiring heavy GPU clusters.

## Results
The pipeline successfully generates high-resolution artistic images by balancing content and style weights. 

*(Note: You can add a 'Before' and 'After' image here in your repo to show off the results!)*

## Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Model Architecture:** VGG19
* **Image Processing:** PIL (Python Imaging Library), Matplotlib
* **Optimization:** L-BFGS / Adam Optimizers

## How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/NisargV2002/](https://github.com/NisargV2002/)[Repo-Name].git
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the style transfer script:
    ```bash
    python style_transfer.py --content "path/to/photo.jpg" --style "path/to/painting.jpg"
    ```
