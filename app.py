import streamlit as st
import requests
import onnxruntime as ort
import pandas as pd
import numpy as np
import tempfile

st.title("ONNX AI Demo from GitHub")

# Raw URL GitHub
onnx_url = "https://raw.githubusercontent.com/AustraliaSilver/BDAI/main/model.onnx"

# Tải mô hình từ GitHub
r = requests.get(onnx_url)
if r.status_code != 200:
    st.error(f"Cannot download ONNX model. Status code: {r.status_code}")
else:
    onnx_bytes = r.content
    st.success("Model downloaded from GitHub!")

    # Lưu vào file tạm và load ONNX
    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_file:
        tmp_file.write(onnx_bytes)
        tmp_file.flush()
        session = ort.InferenceSession(tmp_file.name)

    st.success("Model loaded!")

    # Thông tin input/output
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    st.write(f"Input: {input_name}, Output: {output_name}")

    # Upload dữ liệu CSV
    data_file = st.file_uploader("Upload CSV input data", type="csv")
    if data_file:
        df = pd.read_csv(data_file)
        st.write("Input preview:")
        st.dataframe(df.head())

        input_data = df.to_numpy().astype(np.float32)
        result = session.run(None, {input_name: input_data})
        st.write("Model output:")
        st.write(result[0])
