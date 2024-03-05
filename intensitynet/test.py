import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from torch.autograd import Variable
import time

from model import IntensityNet


inference_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load the trained model
model = IntensityNet(torch.device("cuda:0"))
model.load_state_dict(torch.load('haze_detection_model2.pth'))
model.eval()  # Set the model to evaluation mode
print("-----MODEL LOADED-----")

t = time.time()
# Specify the folder containing the test images
test_folder = 'test'

# Create a list to store the results
results = []

# Iterate over the images in the test folder
for image_path in Path(test_folder).glob('*.*'):  # Assuming the images are in JPEG format
    # Load and preprocess the image
    image = Image.open(image_path)
    input_tensor = inference_transform(image).cuda()
    input_batch = Variable(input_tensor.unsqueeze(0))  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Convert the output to a probability (0 to 1)
    probability = output.item()
    
    # Classify as 'haze' if probability is greater than a threshold (e.g., 0.5)
    classification = 'light' if probability > 0.5 else 'heavy'

    # Store the result
    results.append({'image_path': str(image_path), 'probability': probability, 'classification': classification})

# Display the results
for result in results:
    print(f"Image: {result['image_path']}, Probability: {result['probability']:.4f}, Classification: {result['classification']}")

print(time.time()-t)