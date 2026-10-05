import streamlit as st
import cv2
from src.config import load_config, ROOT
from src.vision.camera import Camera
from src.vision.face_detector import FaceDetector
from src.vision.preprocess import preprocess_face
from src.models.expression_classifier import ExpressionClassifier, ModelUnavailableError
from src.analysis.smoothing import PredictionSmoother
from src.llm.client import LLMClient, LLMError

st.set_page_config(page_title="Biểu cảm quan sát được", page_icon="👁️", layout="wide")
st.title("Phân tích biểu cảm quan sát được")
st.warning("Kết quả là dự đoán biểu cảm từ hình ảnh, không phải kết luận về cảm xúc hoặc chẩn đoán.")
st.caption("Chỉ dùng để tham khảo. Không dùng cho tuyển dụng, chấm điểm, giám sát, đánh giá hoặc quyết định quan trọng.")
config = load_config()
model_cfg = config["model"]

@st.cache_resource
def detector():
    return FaceDetector(model_cfg["face_scale_factor"], model_cfg["min_neighbors"])

if "camera" not in st.session_state:
    st.session_state.camera = None
if "running" not in st.session_state:
    st.session_state.running = False
if "smoother" not in st.session_state:
    st.session_state.smoother = PredictionSmoother(config["labels"], config["smoothing"]["window_size"])
if "llm_result" not in st.session_state:
    st.session_state.llm_result = ""
if "llm_client" not in st.session_state:
    st.session_state.llm_client = LLMClient(config["llm"]["endpoint_env"], config["llm"]["api_key_env"], config["llm"]["model_env"], config["llm"]["min_call_interval_seconds"])

left, right = st.columns([2, 1])
with right:
    model_path = ROOT / model_cfg["path"]
    st.write("**Trạng thái**")
    st.write("Camera:", "Đang chạy" if st.session_state.running else "Đã dừng")
    st.write("Model:", "Sẵn sàng" if model_path.is_file() else "Chưa cài — chế độ demo (không dự đoán)")
    llm_on = st.toggle("Bật diễn giải LLM", value=False, help="Chỉ gửi nhãn và thống kê tổng hợp, không gửi ảnh/video.")
    if st.button("Start camera", disabled=st.session_state.running, use_container_width=True):
        camera = Camera()
        if camera.open():
            st.session_state.camera = camera
            st.session_state.running = True
            st.session_state.smoother.reset()
        else:
            st.error("Không mở được webcam. Kiểm tra quyền truy cập và camera đang được ứng dụng khác sử dụng.")
    if st.button("Stop camera", disabled=not st.session_state.running, use_container_width=True):
        st.session_state.camera.release()
        st.session_state.camera = None
        st.session_state.running = False
        st.session_state.smoother.reset()
        st.rerun()
    result_box = st.empty()
    status_box = st.empty()
    st.caption("Ảnh chỉ được xử lý trong bộ nhớ; ứng dụng không ghi hình hoặc lưu lịch sử.")
    if not model_path.is_file():
        st.info(f"Đặt model Keras tại `{model_path}` để bật inference thực. Xem README để biết yêu cầu input và mapping nhãn.")

with left:
    video = st.empty()
    if st.session_state.running:
        if hasattr(st, "fragment"):
            @st.fragment(run_every=0.12)
            def camera_view():
                camera = st.session_state.camera
                if camera is None:
                    return
                ok, frame = camera.read()
                if not ok:
                    st.error("Không đọc được khung hình. Camera đã được dừng.")
                    camera.release(); st.session_state.camera = None; st.session_state.running = False
                    return
                try:
                    found = detector().detect(frame)
                    display = frame.copy()
                    message = found.message
                    if len(found.faces) == 1:
                        x, y, w, h = found.faces[0]
                        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 220, 80), 2)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        crop = gray[y:y+h, x:x+w]
                        face_bgr = frame[y:y+h, x:x+w]
                        if message == "Đã phát hiện một khuôn mặt." and model_path.is_file():
                            classifier = st.session_state.get("classifier")
                            if classifier is None:
                                classifier = ExpressionClassifier(model_path, config["labels"], model_cfg["input_size"], model_cfg["color_mode"], model_cfg["confidence_threshold"])
                                st.session_state.classifier = classifier
                            pred = classifier.predict(preprocess_face(face_bgr, model_cfg["input_size"], model_cfg["color_mode"]))
                            smooth = st.session_state.smoother.update(pred["scores"])
                            st.session_state.latest_summary = smooth["scores"]
                            if smooth["confidence"] >= model_cfg["confidence_threshold"]:
                                result_box.metric("Biểu cảm được dự đoán (đã làm mượt)", smooth["label"], f"{smooth['confidence']:.0%}")
                            else:
                                result_box.info(f"Chưa đủ chắc chắn ({smooth['confidence']:.0%}).")
                        elif not model_path.is_file():
                            result_box.info("Chế độ demo: đã phát hiện mặt, chưa có model nên không đưa ra dự đoán.")
                        else:
                            result_box.info(message)
                    else:
                        st.session_state.smoother.reset()
                        result_box.info(message)
                    status_box.caption(message)
                    video.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                except ModelUnavailableError as exc:
                    result_box.error(str(exc))
                except Exception as exc:
                    result_box.error(f"Lỗi xử lý: {exc}")
            camera_view()
        else:
            video.info("Webcam đang dừng.")
    else:
        video.info("Nhấn Start camera để bắt đầu. Khi nhấn Stop, camera sẽ được giải phóng.")

if llm_on:
    st.subheader("Diễn giải tổng hợp tùy chọn")
    st.caption("LLM chỉ nhận thống kê chữ do bạn gửi bằng nút bên dưới. Không có ảnh/video nào được gửi.")
    summary = st.session_state.get("latest_summary")
    if summary:
        st.caption("Các điểm số tổng hợp sẽ gửi: " + ", ".join(f"{label} {score:.0%}" for label, score in summary.items()))
    if st.button("Diễn giải kết quả gần đây", disabled=not summary):
        try:
            st.session_state.llm_result = st.session_state.llm_client.interpret(summary)
        except LLMError as exc:
            st.error(str(exc))
    if st.session_state.llm_result:
        st.write(st.session_state.llm_result)
