import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from sklearn.metrics import mean_squared_error

# --- Cấu hình giao diện ---
st.set_page_config(page_title="Demo Tiểu luận Nhóm - Đề tài 6", layout="wide")

# --- Phần 1: Tiêu đề và Mục tiêu ---
st.title("Phần 3: Sản phẩm Demo - Đề tài 6")
st.markdown("""
**Đề tài:** Sự lan truyền sai số trong tính toán số và ảnh hưởng của nhiễu dữ liệu (Noise) đến mô hình xấp xỉ hàm. [cite: 3]
""")

# --- Sidebar: Các thanh trượt điều khiển ---
st.sidebar.header("Cấu hình Thực nghiệm")
st.sidebar.markdown("Thay đổi các thông số dưới đây để thấy sự lan truyền sai số: [cite: 40]")

# 1. Điều chỉnh biên độ nhiễu (Variance)
noise_amplitude = st.sidebar.slider("Biên độ nhiễu (Noise Amplitude):", 0.0, 1.0, 0.2, 0.05)

# 2. Điều chỉnh bậc của đa thức xấp xỉ (Bình phương bé nhất)
poly_degree = st.sidebar.slider("Bậc đa thức Bình phương bé nhất:", 1, 10, 3)

# 3. Nút để tạo lại nhiễu mới
if st.sidebar.button("Sinh lại nhiễu ngẫu nhiên"):
    st.cache_data.clear()

# --- Bước 1: Khởi tạo dữ liệu gốc ---
def real_function(x):
    return np.sin(2 * np.pi * x) + 0.5 * x

x_data = np.linspace(0, 2, 15)
y_true = real_function(x_data)

# --- Bước 2: Tiêm sai số đo lường ---
np.random.seed(42)
noise = np.random.normal(0, noise_amplitude, 15)
y_noisy = y_true + noise

# --- Bước 3: Chạy mô hình ---
# 1. Nội suy Lagrange
lagrange_poly = lagrange(x_data, y_noisy)

# 2. Bình phương bé nhất
ls_coeffs = np.polyfit(x_data, y_noisy, poly_degree)
ls_poly = np.poly1d(ls_coeffs)

# Dữ liệu mượt để vẽ
x_plot = np.linspace(0, 2, 200)
y_plot_true = real_function(x_plot)
y_plot_lagrange = lagrange_poly(x_plot)
y_plot_ls = ls_poly(x_plot)

# --- Bước 4: Trực quan hóa và MSE ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Đồ thị mô phỏng sự lan truyền sai số [cite: 33]")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(x_data, y_noisy, color='red', label='Dữ liệu nhiễu (Noisy Data)', zorder=5)
    ax.plot(x_plot, y_plot_true, 'k-', label='Hàm gốc (Ground Truth)', linewidth=2)
    ax.plot(x_plot, y_plot_lagrange, 'g--', label='Nội suy Lagrange (Bậc 14)')
    ax.plot(x_plot, y_plot_ls, 'b-', label=f'Bình phương bé nhất (Bậc {poly_degree})', linewidth=2)

    ax.set_ylim(min(y_plot_true) - 1.5, max(y_plot_true) + 1.5)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    st.pyplot(fig)

with col2:
    st.subheader("Chỉ số MSE ")
    mse_lagrange = mean_squared_error(y_plot_true, y_plot_lagrange)
    mse_ls = mean_squared_error(y_plot_true, y_plot_ls)

    st.metric("MSE Lagrange", f"{mse_lagrange:.4f}")
    st.metric("MSE Least Squares", f"{mse_ls:.4f}")

    st.markdown("---")
    st.write("**Nhận xét nhanh:**")
    if mse_lagrange > mse_ls:
        st.success("Mô hình Bình phương bé nhất ổn định hơn trước nhiễu.")
    else:
        st.warning("Cảnh báo: Hiện tượng quá khớp (Overfitting) có thể xảy ra.")

st.divider()
expander = st.expander("Xem giải thích 3 câu hỏi định hướng (Phần 2 - Bước 4) [cite: 34-38]")
with expander:
    st.markdown("""
    1. **Tại sao đường Lagrange dao động biên độ lớn?** [cite: 35]  
       - Do nội suy Lagrange bắt buộc phải đi qua chính xác các điểm nhiễu. Với đa thức bậc cao (n=14), các sai số nhỏ tại điểm nút bị phóng đại ở khoảng giữa các điểm, gây ra hiện tượng dao động mạnh (Runge's phenomenon). [cite: 11, 46]
    
    2. **Đường bình phương bé nhất có đi qua các điểm nhiễu không?** [cite: 36]  
       - Không nhất thiết. Nó tìm cách tối thiểu hóa tổng bình phương khoảng cách từ các điểm đến đường cong, tạo ra một đường "trung bình" mịn hơn, do đó ít bị ảnh hưởng bởi các điểm nhiễu đơn lẻ. [cite: 37]
    
    3. **Nên chọn phương pháp nào khi dữ liệu chứa nhiễu?** [cite: 38]  
       - Ta nên chọn **Phương pháp Bình phương bé nhất** để mô hình hóa, vì nó có khả năng lọc nhiễu và phản ánh xu hướng thực tế của hàm số tốt hơn là cố gắng khớp mọi điểm sai số. [cite: 43]
    """)

st.caption("Ứng dụng được xây dựng theo tài liệu hướng dẫn Đề tài 6 - Python cho Khoa học dữ liệu. [cite: 1, 7]")
