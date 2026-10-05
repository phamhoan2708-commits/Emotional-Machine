import os
import time
import requests

SYSTEM_PROMPT = ("Bạn chỉ được diễn giải thận trọng các nhãn biểu cảm quan sát được và thống kê tổng hợp. "
                 "Không suy luận cảm xúc thật, tính cách, ý định, sức khỏe tâm thần hay mức độ trung thực. "
                 "Không chẩn đoán. Nêu rõ đây chỉ là tín hiệu hình ảnh.")

class LLMError(RuntimeError):
    pass

class LLMClient:
    def __init__(self, endpoint_env="LLM_API_URL", api_key_env="LLM_API_KEY", model_env="LLM_MODEL", min_interval=30):
        self.endpoint_env, self.api_key_env, self.model_env = endpoint_env, api_key_env, model_env
        self.min_interval = min_interval
        self.last_call = 0.0

    def interpret(self, summary):
        endpoint, key, model = (os.getenv(self.endpoint_env), os.getenv(self.api_key_env), os.getenv(self.model_env))
        if not endpoint or not key or not model:
            raise LLMError("Cần cấu hình LLM_API_URL, LLM_API_KEY và LLM_MODEL trong môi trường.")
        now = time.monotonic()
        if now - self.last_call < self.min_interval:
            raise LLMError(f"Vui lòng đợi {self.min_interval - (now-self.last_call):.0f} giây trước lần gọi tiếp theo.")
        payload = {"model": model, "temperature": 0.2, "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Diễn giải thống kê nhãn sau một cách ngắn gọn, không khẳng định cảm xúc thật: " + str(summary)}]}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            self.last_call = now
            return result
        except Exception as exc:
            raise LLMError("Không thể gọi dịch vụ LLM.") from exc
