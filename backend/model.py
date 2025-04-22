import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

# Labels for the thorax diseases (14 classes)
LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
          "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", 
          "Fibrosis", "Pleural_Thickening", "Hernia"]

# Step 1: Load the custom-trained ResNet50 model
def load_model():
    model = resnet50(pretrained=False)  # Don't load ImageNet weights
    model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS))  # Adjust output layer

     # Load your custom-trained model (resnet50.pth should be saved in the backend directory)
    model.load_state_dict(torch.load('resnet50.pth', map_location=torch.device('cpu')))  # Map to CPU if no GPU available
    model.eval()  # Set to evaluation mode
    return model

# Step 2: Prediction function to handle image preprocessing and prediction
def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        preds = torch.sigmoid(outputs).squeeze().numpy()  # Multi-label predictions
    
    # Step 1: Create a dictionary of label: score
    prediction = {label: float(preds[idx]) for idx, label in enumerate(LABELS)}
    
    # Step 2: Sort by score and return list of (label, score) tuples
    sorted_preds = sorted(prediction.items(), key=lambda x: x[1], reverse=True)

    return sorted_preds  # ✅ This is what result.html expects

