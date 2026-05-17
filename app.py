import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
from gradcam import get_gradcam

model = load_model("model.keras")

st.set_page_config(page_title="Brain Tumor Detection")

st.title("🧠 Brain Tumor Detection System")

uploaded_file = st.file_uploader("Upload MRI", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    img = np.array(image)
    img = cv2.resize(img, (224,224))

    # ✅ IMPORTANT FIX
    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    st.subheader("Prediction")

    if pred > 0.5:
        st.error(f"Tumor ❌ ({pred:.2f})")
    else:
        st.success(f"No Tumor ✅ ({1-pred:.2f})")

    # Graph
    fig, ax = plt.subplots()
    ax.bar(["No Tumor","Tumor"], [1-pred, pred])
    st.pyplot(fig)

    # 🔥 Grad-CAM
    st.subheader("Grad-CAM 🔥")

    heatmap = get_gradcam(model, img)

    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(
        cv2.resize(np.array(image),(224,224)),
        0.6,
        heatmap,
        0.4,
        0
    )

    st.image(superimposed)