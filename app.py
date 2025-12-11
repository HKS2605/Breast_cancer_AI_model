import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- CONFIGURATION ---
MODEL_PATH = "models/best_model.pth"
CLASS_NAMES = ['Normal', 'Benign', 'Malignant']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. MODEL BUILDER (Same as train.py) ---
def build_model(num_classes=3):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model

# --- 2. LOAD MODEL (Cached for speed) ---
@st.cache_resource
def load_model():
    model = build_model()
    # Load the weights on CPU first to avoid memory errors, then move to Device
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# --- 3. PREPROCESSING ---
def process_image(image):
    # Resize to 256x256 (Same as training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# --- 4. GRAD-CAM ENGINE ---
def generate_heatmap(model, input_tensor, original_image):
    # Target the last convolutional layer of EfficientNet-B0
    # This is usually 'model.features[-1]'
    target_layers = [model.features[-1]]
    
    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # We need the image in a specific format for Grad-CAM (H, W, C) and float32
    # Since we already have the tensor, Grad-CAM handles the forward pass
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    # Resize original image to 256x256 for overlay
    rgb_img = np.array(original_image.resize((256, 256)))
    rgb_img = np.float32(rgb_img) / 255
    
    # Create the overlay
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization

# --- 5. MAIN APP UI ---
st.title("🧬 Breast Cancer Diagnostic Assistant")
st.write("Upload an ultrasound image to generate a diagnosis and XAI heatmap.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display Original Image
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Original Image', use_column_width=True)
        
    # Run Analysis
    if st.button("Analyze Image"):
        with st.spinner('Analyzing...'):
            # 1. Load Model
            model = load_model()
            
            # 2. Preprocess
            input_tensor = process_image(image)
            
            # 3. Get Prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
            
            prediction_label = CLASS_NAMES[pred.item()]
            confidence_score = conf.item() * 100
            
            # 4. Generate Heatmap (XAI)
            heatmap_img = generate_heatmap(model, input_tensor, image)
            
            # 5. Display Results
            with col2:
                st.image(heatmap_img, caption='AI Attention Heatmap (Grad-CAM)', use_column_width=True)
            
            # Metrics
            if prediction_label == "Malignant":
                st.error(f"**Prediction: {prediction_label}**")
            elif prediction_label == "Benign":
                st.success(f"**Prediction: {prediction_label}**")
            else:
                st.info(f"**Prediction: {prediction_label}**")
                
            st.metric(label="Confidence Score", value=f"{confidence_score:.2f}%")
            
            st.info("ℹ️ The Red areas in the heatmap show where the AI focused to make this decision.")