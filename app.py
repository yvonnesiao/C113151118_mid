import streamlit as st
import cv2
import numpy as np
import pandas as pd
import math
import os

st.set_page_config(page_title="AI 影像處理與人臉分析工具", layout="wide")
st.title("互動式影像處理與人臉分析工具")

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * math.log10((255.0 ** 2) / mse)

def plot_histogram(image, channel_name):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    st.line_chart(pd.DataFrame(hist, columns=[f'{channel_name} Pixel Count']))

st.sidebar.header("影像載入")
uploaded_file = st.sidebar.file_uploader("Support PNG, JPG, BMP", type=['png', 'jpg', 'jpeg', 'bmp'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    height, width, channels = img_bgr.shape
    st.sidebar.markdown("### 影像基本資訊")
    st.sidebar.info(f"- 解析度: {width} x {height} px\n- 色彩維度: {channels} Channels\n- 檔案大小: {len(file_bytes) / 1024:.2f} KB")

    tab1, tab2, tab3 = st.tabs(["色彩空間與直方圖", "自訂卷積濾波", "人臉偵測"])

    with tab1:
        color_mode = st.radio("選擇檢視模式", ["Original", "R Channel", "G Channel", "B Channel", "Y Channel", "U Channel", "V Channel"], horizontal=True)
        col_img, col_hist = st.columns(2)
        
        display_img, hist_img = img_rgb.copy(), img_rgb.copy()
        
        if color_mode == "Original":
            hist_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        elif color_mode in ["R Channel", "G Channel", "B Channel"]:
            idx = {"R Channel": 0, "G Channel": 1, "B Channel": 2}[color_mode]
            display_img = np.zeros_like(img_rgb)
            display_img[:, :, idx] = img_rgb[:, :, idx]
            hist_img = img_rgb[:, :, idx]
        else:
            img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
            idx = {"Y Channel": 0, "U Channel": 1, "V Channel": 2}[color_mode]
            display_img = img_yuv[:, :, idx]
            hist_img = img_yuv[:, :, idx]

        col_img.image(display_img, caption=f"預覽: {color_mode}", width='stretch')
        with col_hist:
            st.markdown(f"**{color_mode} 強度分布直方圖**")
            plot_histogram(hist_img, color_mode)

    with tab2:
        col_matrix, col_result = st.columns([1, 2])
        with col_matrix:
            matrix_str = st.text_area("Kernel Matrix (輸入二維陣列)", "0, -1, 0\n-1, 5, -1\n0, -1, 0", height=150)
            normalize = st.checkbox("進行權重正規化", value=False)
            
            try:
                lines = matrix_str.strip().split('\n')
                kernel = np.array([[float(x.strip()) for x in line.split(',')] for line in lines])
                
                if kernel.shape[0] != kernel.shape[1]:
                    st.error("請輸入 N x N 的方陣！")
                    kernel = None
                else:
                    if normalize and np.sum(kernel) != 0:
                        kernel = kernel / np.sum(kernel)
            except:
                st.error("矩陣格式錯誤")
                kernel = None

        with col_result:
            if kernel is not None:
                img_filtered = cv2.filter2D(img_rgb, -1, kernel)
                psnr_value = calculate_psnr(img_rgb, img_filtered)
                
                st.metric(label="PSNR (峰值信噪比)", value=f"{psnr_value:.2f} dB" if psnr_value != float('inf') else "∞ (完全相同)")
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="原圖", width='stretch')
                c2.image(img_filtered, caption="濾波後", width='stretch')

    with tab3:
        st.write("目前使用演算法：OpenCV Haar-Cascade")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            scale_factor = st.slider("縮放因子 (Scale Factor)", 1.01, 1.5, 1.1, 0.01)
        with col_p2:
            min_neighbors = st.slider("最小相鄰矩形 (Min Neighbors)", 1, 15, 5)
        
        cascade_filename = "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_filename):
            import urllib.request
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, cascade_filename)
            
        face_cascade = cv2.CascadeClassifier(cascade_filename)
        
        if face_cascade.empty():
            st.error("模型載入失敗，請確認網路連線或稍後重試。")
        else:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            
            faces, rejectLevels, levelWeights = face_cascade.detectMultiScale3(
                gray, 
                scaleFactor=scale_factor, 
                minNeighbors=min_neighbors, 
                outputRejectLevels=True
            )
            
            img_faces = img_rgb.copy()
            face_count = len(faces) if len(faces) > 0 else 0
            
            if face_count > 0:
                for (x, y, w, h), weight in zip(faces, levelWeights):
                    # 修正：相容不同 OpenCV 版本的權重格式
                    w_val = np.ravel(weight)[0]
                    confidence = min(float(w_val) / 5.0, 1.0) 
                    
                    cv2.rectangle(img_faces, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(img_faces, f"Conf: {confidence:.2f}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            st.success(f"偵測到 **{face_count}** 張人臉")
            st.image(img_faces, width='stretch')
else:
    st.info("請從左側上傳影像以開始分析。")