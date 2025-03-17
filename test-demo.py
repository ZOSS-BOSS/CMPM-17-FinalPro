import torch
from torchvision import transforms
from PIL import Image
from animal import AnimalCNN


model = AnimalCNN(num_classes=10)  # Replace 10 with your actual number of classes
model.load_state_dict(torch.load("animal_cnn_model.pth"))
model.eval()  # Set to evaluation mode


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
    
])
 
img_path = "CMPM-17-FinalPro/Animal Image Dataset/elephant/1449.jpg"
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0)  


with torch.no_grad():
    outputs = model(image) 
    predicted_class_idx = torch.argmax(outputs).item()


print(f"Predicted class index: {predicted_class_idx}")
