import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import cv2
import io
import zipfile
import random

st.set_page_config(
    page_title="Image Augmentation Studio",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Data Augmentation Studio")
st.markdown("Interactive Preview aur **Batch Dataset Generator (Up to 50 Variations)**.")

mode = st.sidebar.radio("📌 Select Mode", ["1. Live Single Preview", "2. Batch Multi-Image Generator (ZIP)"])

# Zoom In / Zoom Out function
def apply_zoom(img, factor):
    w, h = img.size
    if factor == 1.0:
        return img
    elif factor > 1.0:
        # Zoom In: Center crop aur wapas scale up
        crop_w = int(w / factor)
        crop_h = int(h / factor)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        return img.crop((left, top, left + crop_w, top + crop_h)).resize((w, h), Image.Resampling.LANCZOS)
    else:
        # Zoom Out: Scale down aur canvas par center fit (with black borders)
        new_w = int(w * factor)
        new_h = int(h * factor)
        resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        pad_x = (w - new_w) // 2
        pad_y = (h - new_h) // 2
        padded_img = Image.new("RGB", (w, h), (0, 0, 0))
        padded_img.paste(resized_img, (pad_x, pad_y))
        return padded_img

# Helper function for random batch transformations
def apply_random_augmentation(pil_img):
    img = pil_img.copy()
    
    # 1. Random Flip
    if random.choice([True, False]):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.choice([True, False]):
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
    # 2. Random Rotation (-60 to 60 deg)
    angle = random.randint(-60, 60)
    img = img.rotate(angle, expand=False)
    
    # 3. Random Zoom In / Zoom Out (0.7 to 1.6)
    zoom_val = round(random.uniform(0.7, 1.6), 2)
    img = apply_zoom(img, zoom_val)
    
    # 4. Random Brightness & Contrast
    b_val = random.uniform(0.6, 1.4)
    img = ImageEnhance.Brightness(img).enhance(b_val)
    
    c_val = random.uniform(0.6, 1.4)
    img = ImageEnhance.Contrast(img).enhance(c_val)
    
    # 5. Gaussian Blur (50% chance)
    if random.choice([True, False]):
        np_img = np.array(img)
        k_size = random.choice([3, 5])
        np_img = cv2.GaussianBlur(np_img, (k_size, k_size), 0)
        img = Image.fromarray(np_img)
        
    # 6. Gaussian Noise (40% chance)
    if random.choice([True, False, False]):
        np_img = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, random.randint(8, 20), np_img.shape)
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(np_img)
        
    return img

uploaded_file = st.file_uploader("Upload an Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_img = Image.open(uploaded_file).convert("RGB")

    # ==========================================
    # MODE 1: SINGLE PREVIEW
    # ==========================================
    if mode == "1. Live Single Preview":
        st.sidebar.subheader("⚙️ Custom Controls")
        flip_h = st.sidebar.checkbox("Horizontal Flip")
        flip_v = st.sidebar.checkbox("Vertical Flip")
        rot = st.sidebar.slider("Rotation Angle", -180, 180, 0, step=5)
        
        # Zoom Slider: < 1.0 Zoom Out, > 1.0 Zoom In
        zoom = st.sidebar.slider("Zoom Level (< 1.0 = Zoom Out, > 1.0 = Zoom In)", min_value=0.5, max_value=2.0, value=1.0, step=0.05)
        
        bright = st.sidebar.slider("Brightness", 0.2, 2.0, 1.0, step=0.1)
        cont = st.sidebar.slider("Contrast", 0.2, 2.0, 1.0, step=0.1)
        blur = st.sidebar.checkbox("Apply Blur")
        
        aug_img = original_img.copy()
        if flip_h: aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
        if flip_v: aug_img = aug_img.transpose(Image.FLIP_TOP_BOTTOM)
        if rot != 0: aug_img = aug_img.rotate(rot, expand=False)
        
        # Zoom apply
        aug_img = apply_zoom(aug_img, zoom)
            
        if bright != 1.0: aug_img = ImageEnhance.Brightness(aug_img).enhance(bright)
        if cont != 1.0: aug_img = ImageEnhance.Contrast(aug_img).enhance(cont)
        
        if blur:
            np_arr = cv2.GaussianBlur(np.array(aug_img), (5, 5), 0)
            aug_img = Image.fromarray(np_arr)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Original")
            st.image(original_img, use_container_width=True)
        with col2:
            st.subheader("✨ Augmented")
            st.image(aug_img, use_container_width=True)

        buf = io.BytesIO()
        aug_img.save(buf, format="PNG")
        st.sidebar.download_button("📥 Download Single Image", buf.getvalue(), "augmented_image.png", "image/png")

    # ==========================================
    # MODE 2: BATCH GENERATOR (UP TO 50 IMAGES)
    # ==========================================
    else:
        st.subheader("📦 Generate Up to 50 Variations from One Image")
        num_variations = st.slider("Select variations count", min_value=5, max_value=50, value=50, step=5)

        if st.button("🚀 Generate All Variations"):
            generated_images = []
            
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            for i in range(num_variations):
                new_img = apply_random_augmentation(original_img)
                generated_images.append(new_img)
                progress_bar.progress((i + 1) / num_variations)
                progress_text.text(f"Generating variation {i + 1} of {num_variations}...")
            
            progress_text.empty()
            st.success(f"Generated {num_variations} unique augmented images!")

            # ZIP create karein
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx, img in enumerate(generated_images):
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format="PNG")
                    zip_file.writestr(f"augmented_image_{idx+1}.png", img_byte_arr.getvalue())

            # Top par hi ZIP download button
            st.download_button(
                label=f"📦 Download All {num_variations} Images (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="augmented_dataset_50.zip",
                mime="application/zip"
            )

            st.markdown("---")
            st.subheader("👀 Preview of Generated Images")
            # 4 images per row grid
            cols = st.columns(4)
            for idx, img in enumerate(generated_images):
                with cols[idx % 4]:
                    st.image(img, caption=f"Variation #{idx+1}", use_container_width=True)

else:
    st.info("👆 Pehle image upload karein.")