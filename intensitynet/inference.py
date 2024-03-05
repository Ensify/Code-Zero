import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from intensitynet.model import IntensityNet

inference_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the trained model
model = IntensityNet(device)
model.load_state_dict(torch.load("weights\haze_detection_model.pth", map_location= device))
model.eval()  # Set the model to evaluation mode
print("-----INTENSITY NET MODEL LOADED-----")

def classify_intensity(image):
    # image = Image.open(image_path)
    input_tensor = inference_transform(image).cuda()
    input_batch = Variable(input_tensor.unsqueeze(0))

    with torch.no_grad():
        output = model(input_batch)

    # Convert the output to a probability (0 to 1)
    probability = output.item()
    
    # Classify as 'haze' if probability is greater than a threshold (e.g., 0.5)
    classification = 'light' if probability > 0.5 else 'heavy'

    return classification
