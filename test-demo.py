import torch
from animal import AnimalCNN

model = AnimalCNN()

model.load_state_dict (torch.load("animal_cnn_model.pth", weights_only= True))
guess = model("CMPM-17-FinalPro\Animal Image Dataset\elephant\1449.jpg")
print(guess)