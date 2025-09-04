import streamlit as st
import onnxruntime as ort
import pandas as pd
import numpy as np

st.title("AI ONNX Inference App")

# 1. Upload mô hình ONNX
onnx_file = st.file_uploader("Upload ONNX model", type="onnx")
if onnx_file:
    st.success("Model uploaded successfully!")
    session = ort.InferenceSession(onnx_file.read())
    st.write("Model loaded!")

    # Hiển thị thông tin input/output của mô hình
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    st.write(f"Input name: {input_name}, shape: {input_shape}")
    output_name = session.get_outputs()[0].name
    st.write(f"Output name: {output_name}")

    # 2. Upload dữ liệu đầu vào
    data_file = st.file_uploader("Upload input data (CSV)", type="csv")
    if data_file:
        st.success("Data uploaded successfully!")
        df = pd.read_csv(data_file)
        st.write("Input data preview:")
        st.dataframe(df.head())

        # 3. Chạy mô hình
        try:
            # Convert dataframe thành numpy array
            input_data = df.to_numpy().astype(np.float32)
            
            # Nếu mô hình cần batch dimension
            if len(input_shape) == 2 and input_shape[0] == "None":
                # đảm bảo shape đúng (batch_size, features)
                pass
            
            # Chạy mô hình
            result = session.run(None, {input_name: input_data})
            
            # 4. Hiển thị kết quả
            st.write("Model output:")
            st.write(result[0])
        except Exception as e:
            st.error(f"Error during inference: {e}")
