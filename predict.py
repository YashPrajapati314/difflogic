import torch
from experiments.main_conv import MNISTArchitecture
from PIL import Image
import torchvision.transforms as transforms

def load_model(model_path, model_scale=16, temperature=6.5):
    """
    Loads a trained MNISTArchitecture model from a saved state dictionary.
    """
    # 1. Instantiate the model with the same architecture as when it was trained
    model = MNISTArchitecture(model_scale=model_scale, temperature=temperature)
    
    # 2. Load the saved weights from the file
    model.load_state_dict(torch.load(model_path))
    
    # 3. Set the model to evaluation mode (this is important!)
    model.eval()
    
    return model

def prepare_custom_image(image_path):
    """
    Loads a custom image, converts it to the format expected by the model.
    """
    # Open the image and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # Define the same transformations used for MNIST (resize, to tensor)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    
    # Apply transformations
    img_tensor = transform(img)
    
    # The model expects a batch, so add a batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    # Binarize the image just like in the training script
    img_tensor = torch.where(img_tensor != 0, torch.ones_like(img_tensor), torch.zeros_like(img_tensor))
    
    return img_tensor

if __name__ == '__main__':
    # --- Step 1: Train and Save Your Model ---
    # First, you need to modify your training script ('experiments/main_conv.py')
    # to save the model after training. Add this line at the end of the 'main' function:
    #
    # torch.save(model.state_dict(), "mnist_conv_model.pth")
    #
    # Now, run your training. A file named "mnist_conv_model.pth" will be created.

    # --- Step 2: Load the Trained Model ---
    MODEL_FILE_PATH = "mnist_conv_model.pth"
    print(f"Loading model from {MODEL_FILE_PATH}...")
    trained_model = load_model(MODEL_FILE_PATH)
    print("Model loaded successfully!")

    # --- Step 3: Prepare Your Custom Input ---
    # Create a dummy image or use your own. For this example, we'll create a
    # random 28x28 black and white image.
    # Replace this with `prepare_custom_image('path/to/your/image.png')` for a real image.
    custom_input_tensor = torch.randint(0, 2, (1, 1, 28, 28)).float()
    print(f"\nUsing a random 28x28 tensor as input.")

    # --- Step 4: Make a Prediction ---
    with torch.no_grad(): # Disable gradient calculation for inference
        output = trained_model(custom_input_tensor)
        
        # The output of the model is logits. To get the predicted class,
        # we find the index of the highest logit.
        prediction = torch.argmax(output, dim=1)
        
    print(f"\nModel output (logits): {output.numpy().flatten()}")
    print(f"Predicted class: {prediction.item()}")
