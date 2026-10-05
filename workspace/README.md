# MVP phân tích biểu cảm quan sát được

Ứng dụng Python/Streamlit dùng OpenCV để đọc webcam, dò mặt, tiền xử lý và gọi CNN Keras nếu có model tương thích. Mặc định ứng dụng chạy cục bộ, không lưu frame/lịch sử và LLM tắt. Không có model kèm theo; khi thiếu model, giao diện chỉ cho biết đã dò mặt và không bịa dự đoán.

## Giới hạn và cách dùng có trách nhiệm

Ứng dụng chỉ mô tả **biểu cảm quan sát được từ hình ảnh**. Dự đoán không xác nhận cảm xúc thật, tính cách, ý định, sức khỏe tâm thần hay mức độ trung thực và không phải chẩn đoán. Không sử dụng cho tuyển dụng, chấm điểm, giám sát, đánh giá nhân viên/học sinh hoặc quyết định quan trọng. Không có nhận diện danh tính. Thông báo giới hạn luôn hiện trên UI.

Kết quả có thể sai khi thiếu sáng, ảnh mờ, mặt quay nghiêng, đeo kính, bị che khuất, hoặc do khác biệt giữa người dùng và dữ liệu huấn luyện. Haar Cascade chỉ là bộ dò mặt cơ bản, không phải hệ thống theo dõi danh tính. Ngưỡng confidence mặc định 0.60 có thể điều chỉnh, nhưng confidence chưa được hiệu chuẩn và không đại diện xác suất cảm xúc thật. Smoothing lấy trung bình điểm số trong cửa sổ khung hình gần nhất.

## Cài và chạy

Yêu cầu Python 3.10 trở lên. Trên Windows, TensorFlow/OpenCV có thể cần phiên bản Python/driver phù hợp với môi trường máy.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Mở URL Streamlit được in trong terminal và cấp quyền webcam cho ứng dụng/trình duyệt. Nhấn **Start camera**, sau đó **Stop camera** để giải phóng thiết bị. Nếu không mở được camera, đóng ứng dụng khác đang dùng webcam, kiểm tra quyền riêng tư camera của Windows, rồi thử lại. Việc truy cập camera phụ thuộc môi trường chạy Streamlit; nếu chạy từ xa, webcam của máy chủ không phải webcam máy người xem.

## Model và mapping nhãn

Không tải model tự động. Đặt model Keras có license phù hợp tại `models/expression_model.keras`. Model mặc định phải nhận tensor `(N, 48, 48, 1)` grayscale float `[0,1]` và xuất 7 giá trị theo đúng thứ tự `labels` trong `config.yaml`. Thứ tự mặc định là `angry, disgust, fear, happy, sad, surprise, neutral`; nếu model dùng thứ tự khác, sửa danh sách này để khớp chính xác. Nếu model dùng input khác, sửa `input_size` và `color_mode` rồi điều chỉnh preprocessing trong `src/vision/preprocess.py` cho phù hợp. Mã kiểm tra số output so với số nhãn.

Không có model/dataset đi kèm, do đó không đưa ra điểm chính xác nào. FER-2013 là dataset thường dùng cho bảy nhãn này; hãy lấy tài liệu nguồn/license trực tiếp từ nhà cung cấp dataset (Kaggle/ICML challenge tùy bản phân phối) và xác minh quyền sử dụng trước khi tải hoặc dùng thương mại. Model huấn luyện từ FER-2013 kế thừa các giới hạn dữ liệu và không nên xem như đo lường cảm xúc chuẩn. OpenCV Haar Cascade được cung cấp trong bản phân phối OpenCV; xem license và notice đi kèm phiên bản OpenCV bạn cài.

Pipeline inference nằm độc lập trong `src/models/expression_classifier.py`; thay model bằng file Keras tương thích, không cần thay webcam/UI. Dự án chưa kèm script huấn luyện, vì chưa chọn dataset có nguồn/license đã xác minh và cấu trúc nhãn/model.

## LLM tùy chọn

LLM tắt mặc định. Bật công tắc trong UI sẽ mở phần diễn giải. LLM client chỉ nhận chuỗi thống kê chữ do người vận hành nhập; không có đường truyền frame hoặc ảnh. Chỉ bật nếu đã kiểm tra endpoint và chính sách lưu trữ của nhà cung cấp. Cấu hình biến môi trường `LLM_API_URL`, `LLM_API_KEY`, `LLM_MODEL`; không đặt key trong mã nguồn. Ví dụ mẫu nằm trong `.env.example` (không chứa secret). Client áp giới hạn gọi tối thiểu 30 giây và prompt yêu cầu tránh kết luận/diagnosis. Lỗi API không in key. Tắt công toggle để không dùng LLM.

## Quyền riêng tư

Frame được giữ trong bộ nhớ của tiến trình trong lúc xử lý, không ghi file và không lưu lịch sử. Camera chỉ được mở sau khi người dùng nhấn Start và được release khi nhấn Stop hoặc gặp lỗi đọc frame. Không có logging ảnh/video hay dữ liệu nhận dạng cá nhân. Nếu LLM được bật và người dùng bấm gửi thống kê, nhà cung cấp LLM có thể xử lý nội dung theo chính sách của họ.

## Kiểm tra

```powershell
python -m pytest -q
python -m compileall -q app.py src tests
```

Test pipeline không cần webcam hoặc TensorFlow thật; model inference được mock.
