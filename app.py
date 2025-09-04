import streamlit as st
import requests
import onnxruntime as ort
import pandas as pd
import numpy as np
import tempfile

st.title("ONNX AI Demo from GitHub")

# URL raw GitHub của mô hình
onnx_url = "https://raw.githubusercontent.com/australia1/BDAI/main/model.onnx"

# 1. Download mô hình
r = requests.get(onnx_url)
r.raise_for_status()  # đảm bảo tải thành công
onnx_bytes = r.content

# 2. Lưu file tạm và load mô hình
with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_file:
    tmp_file.write(onnx_bytes)
    tmp_file.flush()
    session = ort.InferenceSession(tmp_file.name)

st.success("Model loaded!")

# 3. Thông tin input/output
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
st.write(f"Input: {input_name}, Output: {output_name}")

# 4. Upload CSV
data_file = st.file_uploader("Upload CSV input data", type="csv")
if data_file:
    df = pd.read_csv(data_file)
    st.write("Input preview:")
    st.dataframe(df.head())

    input_data = df.to_numpy().astype(np.float32)
    result = session.run(None, {input_name: input_data})
    st.write("Model output:")
    st.write(result[0])
