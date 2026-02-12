import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from streamlit_option_menu import option_menu
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Lumina Breast AI", layout="wide", page_icon="🧬")

# Path to your best model
MODEL_PATH = "models/best_model.pth"
CLASS_NAMES = ['Normal', 'Benign', 'Malignant']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CUSTOM CSS (The "Hack" for the Dashboard Look) ---
st.markdown("""
    <style>
        /* Main Background */
        .stApp {
            background-color: #F8F9FA;
        }
        
        /* Card Styling */
        .metric-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        /* Remove default top padding */
        .block-container {
            padding-top: 2rem;
        }
        
        /* Upload Box Styling */
        .stFileUploader {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# --- MODEL FUNCTIONS (Same as before) ---
def build_model(num_classes=3):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model

@st.cache_resource
def load_model():
    model = build_model()
    # Safely load weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def generate_heatmap(model, input_tensor, original_image):
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    rgb_img = np.array(original_image.resize((256, 256)))
    rgb_img = np.float32(rgb_img) / 255
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization

# --- UI LAYOUT ---

# 1. Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=50) # Placeholder Logo
    st.title("Lumina")
    st.caption("Breast AI Diagnostic Tool")
    
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Patient Database", "Model Stats", "Settings"],
        icons=["grid", "people", "graph-up", "gear"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "black", "font-size": "14px"}, 
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#007bff"},
        }
    )

# 2. Main Content Area
if selected == "Dashboard":
    # Header
    today = datetime.now().strftime("%A, %B %d, %Y")
    st.title("Good Morning, Dr.")
    st.text(f"{today}")
    
    # Metrics Cards (Custom HTML)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin:0; color:#007bff">48</h3>
            <p style="color:gray; margin:0">Total Scans Today</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin:0; color:#28a745">91.0%</h3>
            <p style="color:gray; margin:0">AI Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Upload Section
    st.subheader("Upload New Scan")
    uploaded_file = st.file_uploader("Drag and drop DICOM, PNG, JPEG here", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Load and Display
        image = Image.open(uploaded_file).convert("RGB")
        
        # Create a clean layout for analysis
        col_img, col_res = st.columns([1, 2])
        
        with col_img:
            st.image(image, caption='Uploaded Scan', use_container_width=True)
            analyze_btn = st.button("Run AI Diagnosis", type="primary")
        
        if analyze_btn:
            with st.spinner("Lumina AI is analyzing the scan..."):
                try:
                    # Run Inference
                    model = load_model()
                    input_tensor = process_image(image)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        conf, pred = torch.max(probs, 1)
                        
                    prediction = CLASS_NAMES[pred.item()]
                    confidence = conf.item() * 100
                    
                    # Generate Heatmap
                    heatmap = generate_heatmap(model, input_tensor, image)
                    
                    # Display Results in the right column
                    with col_res:
                        st.markdown(f"### AI Prediction: **{prediction}**")
                        
                        # Dynamic Badge Color
                        if prediction == "Malignant":
                            st.error(f"Confidence: {confidence:.2f}% (Requires Urgent Review)")
                        elif prediction == "Benign":
                            st.success(f"Confidence: {confidence:.2f}%")
                        else:
                            st.info(f"Confidence: {confidence:.2f}%")
                            
                        st.image(heatmap, caption='Grad-CAM Explanation', width=300)
                        
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    # Recent Patient Cases (Mock Data Table)
    st.markdown("---")
    st.subheader("Recent Patient Cases")
    
    # Create fake data to look like your screenshot
    data = {
        "Patient ID": ["PT-4920", "PT-4919", "PT-4918"],
        "Scan Date": ["10:45 AM", "10:30 AM", "09:15 AM"],
        "AI Prediction": ["Malignant", "Benign", "Normal"],
        "Confidence": ["94%", "88%", "99%"],
        "Status": ["Review Required", "Finalized", "Finalized"]
    }
    df = pd.DataFrame(data)
    
    # Custom Styling for the Table
    def style_status(val):
        color = 'red' if val == 'Malignant' else 'green' if val == 'Benign' else 'blue'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        df.style.map(style_status, subset=['AI Prediction']),
        use_container_width=True,
        hide_index=True
    )

elif selected == "Patient Database":
    st.title("Patient Database")
    st.write("Database features would go here...")
    
elif selected == "Model Stats":
    st.title("Model Performance Stats")
    st.write("Accuracy charts would go here...") 