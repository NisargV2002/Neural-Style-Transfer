import torch
import torch.optim as optim
from torchvision import models,transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Image loader function

def load_image(img_path, max_size = 400):
    image = Image.open(img_path)

    # Resize the image

    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor()
    ])

    # Add batch Dimension

    image = transform(image).unsqueeze(0)

    return image

# Load the images

content_image = load_image('/Users/nisargvaishnav/Desktop/NST/nst_env/nst/content.jpg')
style_image = load_image('/Users/nisargvaishnav/Desktop/NST/nst_env/nst/style.jpg')

# content_image = load_image('/Users/nisargvaishnav/Desktop/NST/nst_env/nst/content1.jpg')
# style_image = load_image('/Users/nisargvaishnav/Desktop/NST/nst_env/nst/style1.jpeg')

# Display the images

def imshow(tensor, title=None):
    
    tensor = tensor.clone().detach()  # Remove gradient tracking
    if tensor.dim() == 4:  # If batch dimension is present
        tensor = tensor.squeeze(0)  # Remove batch dimension

    if tensor.shape[0] == 3:  # (C, H, W) format
        image = tensor.numpy().transpose(1, 2, 0)  # Convert to (H, W, C)
    elif tensor.shape[-1] == 3:  # Already (H, W, C)
        image = tensor.numpy()
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    plt.imshow(image)
    if title:
        plt.title(title)

    plt.axis('off') 
    plt.show()

imshow(content_image, 'Content Image')
imshow(style_image, 'Style Image')

# Load the pre trained VGG - 19 Model

vgg = models.vgg19(pretrained = True).features.eval()

# Move to CPU

device = torch.device('cpu')
content_image = content_image.to(device)
style_image = style_image.to(device)
vgg.to(device)

# Define the layers to extract features from

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Function to extract the features from the model

def get_features(image, model):
    features = {}
    x = image

    for name, layer in enumerate(model.children()):
        x = layer(x)

        # Store the content and style features

        layer_name = f'conv_{name}'
        if layer_name in content_layers:
            features['content'] = x
        if layer_name in style_layers:
            features[layer_name] = x

    return features        

# Compute the style Loss

def gram_matrix(tensor):

    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h*w) # FLatten the feature map
    gram = torch.mm(tensor, tensor.t()) # compute the gram matrix
    return gram

# Define the loss function

def compute_loss(content_features, style_features, target_features):
    content_loss = torch.mean((target_features['content'] - content_features['content'])**2)

    style_loss = 0

    for layer in style_layers:
        gram_target = gram_matrix(target_features[layer])
        gram_style = gram_matrix(style_features[layer])
        style_loss += torch.mean((gram_target - gram_style)**2)

    # Weigh the losses

    total_loss = 1e5 * style_loss + content_loss    
    return total_loss

# Optimizing the Image

# Initialize the target image as a copy of the content image

target_img = content_image.clone().requires_grad_(True).to(device)

# Set the optimizer
optimizer = optim.Adam([target_img], lr = 0.003)

# Training the model

# Training loop

epochs = 300

# Compute initial loss (before training)
with torch.no_grad():
    initial_loss = compute_loss(
        get_features(content_image, vgg), 
        get_features(style_image, vgg), 
        get_features(target_img, vgg)
    ).item()

for step in range(epochs):
    optimizer.zero_grad() # Reset Gradients

    # Extract features from target image
    target_features = get_features(target_img, vgg)

    # Compute loss
    loss = compute_loss(get_features(content_image, vgg), get_features(style_image, vgg), target_features)

    loss.backward() # compute Gradients
    optimizer.step() # Update Image

    accuracy = max(0, 100 * (1 - loss.item() / initial_loss))  # Ensure accuracy doesn't go below 0%


    # Display Progress
    if step%50==0:
        print(f'Epoch [{step}] / [{epochs}], Loss : {loss.item():.4f}, Accuracy : {accuracy:.2f}%')
        imshow(target_img, title=f"Epoch {step}")

# Show final output

imshow(target_img, title="Final Styled Image")

# Save the generated output

def save_image(tensor, filename = "output.png"):
    image = tensor.clone().detach().squeeze(0) # Remove Batch Dimension
    image = image.numpy().transpose(1, 2, 0) # Convert to (H, W, C)
    image = (image * 255).astype(np.unint8) # Convert to 8 Bit
    Image.fromarray(image).save(filename)

save_image(target_img, "styled_image.png")
print("Saved final stylized image as 'styled_image.png' ")    






