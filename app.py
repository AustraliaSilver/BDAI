import requests
import onnxruntime as ort
import numpy as np
import pandas as pd
import streamlit as st

st.title("ONNX AI Demo from GitHub")

# 1. URL raw GitHub của mô hình
onnx_url = "https://raw.githubusercontent.com/australia1/BDAI/main/model.onnx"

# Tải mô hình từ GitHub
response = requests.get(onnx_url)
onnx_bytes = response.content
st.success("Model downloaded from GitHub!")

# Load mô hình
session = ort.InferenceSession(onnx_bytes)
st.write("Model loaded!")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
st.write(f"Input: {input_name}, Output: {output_name}")

# 2. Upload dữ liệu CSV từ người dùng
data_file = st.file_uploader("Upload CSV input data", type="csv")
if data_file:
    df = pd.read_csv(data_file)
    st.write("Input preview:")
    st.dataframe(df.head())

    # 3. Chạy mô hình
    input_data = df.to_numpy().astype(np.float32)
    result = session.run(None, {input_name: input_data})
    st.write("Model output:")
    st.write(result[0])
