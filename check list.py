import os

# This should match the model filename you placed in backend
model_filename = "resnet50.pth"
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_filename)

print(f"Checking model path: {model_path}")

# Check if file exists
if os.path.isfile(model_path):
    print("✅ Model file found!")
else:
    print("❌ Model file NOT found. Check the filename or location.")
