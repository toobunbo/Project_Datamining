## Bản Kế hoạch: Dự án Data Mining
**Tên dự án:** "Dự đoán Sớm Nguy cơ Nhập ICU cho Bệnh nhân COVID-19"


## 1. Bối cảnh & Vấn đề 

Trong đại dịch COVID-19, một trong những thách thức lớn nhất là sự quá tải của hệ thống y tế, đặc biệt là các Đơn vị Chăm sóc Tích cực (ICU). Nguồn lực (giường bệnh, máy thở, nhân sự) là hữu hạn.

Các bác sĩ tại phòng cấp cứu phải ra quyết định nhanh chóng: Bệnh nhân A này có nguy cơ trở nặng và cần ICU hay không?

**Mục tiêu của chúng ta:** Xây dựng một mô hình Data Mining (Phân loại) để hỗ trợ bác sĩ ra quyết định. Mô hình sẽ dự đoán khả năng một bệnh nhân cần nhập ICU chỉ dựa trên các dữ liệu lâm sàng thu thập được trong **2 giờ đầu** tiên họ nhập viện.

## 2. Mục tiêu Dự án

Chúng ta chia mục tiêu thành 3 cấp độ rõ ràng:

### (A) Mục tiêu Thực tiễn (Y tế)
* Hỗ trợ bác sĩ ưu tiên phân bổ nguồn lực ICU.
* Cảnh báo sớm các ca có nguy cơ trở nặng.
* Giúp nhận diện các ca nguy cơ thấp để theo dõi ở khu vực thường, giảm tải cho ICU.

### (B) Mục tiêu Kỹ thuật (Data Mining)
* Xây dựng và so sánh ít nhất 2 mô hình Phân loại Nhị phân:
    1.  **Logistic Regression** (Ưu tiên khả năng giải thích).
    2.  **Random Forest** (Ưu tiên độ chính xác).
* Đánh giá mô hình bằng các chỉ số: Confusion Matrix, ROC-AUC, F1-Score.
* **Quan trọng nhất:** Tối ưu chỉ số **Recall của lớp "Cần ICU" (y=1)**. Chúng ta chấp nhận dự đoán nhầm (False Positive) còn hơn bỏ sót ca nặng (False Negative).
* Xử lý 2 thách thức kỹ thuật lớn:
    * **Dữ liệu Mất cân bằng** (Imbalanced Data): Số ca $y=1$ (cần ICU) chắc chắn sẽ ít hơn $y=0$. (Phương án: SMOTE hoặc `class_weight`).
    * **Dữ liệu Thiếu** (Missing Data): Dữ liệu y tế luôn bị thiếu. (Phương án: Imputation).

### (C) Mục tiêu Phân tích 
* Không chỉ dự đoán, mà phải **giải thích** *tại sao*.
* Xác định các yếu tố (features) quan trọng nhất (ví dụ: SpO2, Tuổi, xét nghiệm CRP...) bằng Feature Importance.
* Sử dụng **SHAP** để giải thích mô hình ở 2 cấp độ:
    * **Global:** Yếu tố nào làm tăng/giảm nguy cơ ICU trên toàn bộ bệnh nhân?
    * **Local:** Phân tích 1-2 bệnh nhân cụ thể: "Tại sao mô hình dự đoán ca này cần ICU?"
### 3. Dữ liệu & Phạm vi (Input / Output)

#### Dataset Nguồn
* Tập trung vào dataset "Kaggle: Sírio-Libanês ICU Prediction". Đây là bộ dữ liệu khớp hoàn hảo với mục tiêu của chúng ta.

#### Lọc & Tiền xử lý 
1.  **Lọc theo Thời gian:** Chỉ giữ lại các đặc trưng (dấu hiệu sinh tồn, xét nghiệm máu) được thu thập trong "cửa sổ" 0-2 giờ đầu tiên.
2.  **Loại bỏ Rò rỉ Dữ liệu:** Loại bỏ những bệnh nhân đã được nhập ICU ngay từ đầu (ví dụ: được chuyển thẳng đến ICU từ bệnh viện khác).

#### Định nghĩa Đầu vào (Input - 𝑋)
* Vector đặc trưng của bệnh nhân tại cửa sổ 0-2h, bao gồm:
    * **Nhân khẩu học:** Tuổi, Giới tính.
    * **Bệnh nền:** (Ví dụ: tiểu đường, cao huyết áp, béo phì... nếu có).
    * **Dấu hiệu sinh tồn (0-2h):** SpO2, nhịp tim, huyết áp, nhiệt độ, nhịp thở...
    * **Xét nghiệm máu (0-2h):** Bạch cầu, CRP, Creatinine...

#### Định nghĩa Đầu ra (Output - 𝑦)
* Đây là bài toán **Phân loại Nhị phân**. Nhóm 1 sẽ phải tạo ra cột `y` này.
* `y = 1` (Cần ICU): Nếu bệnh nhân **có** nhập ICU ở *bất kỳ thời điểm nào* trong suốt quá trình điều trị (0h-2h).
* `y = 0` (Không cần ICU): Nếu bệnh nhân **không** nhập ICU và được xuất viện hoặc điều trị ở khu vực thường.

