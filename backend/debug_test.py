import os
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.admin_server import app, startup_event

print("Triggering startup event...")
startup_event()

client = TestClient(app)

print("Creating dummy image...")
from PIL import Image
import io
img = Image.new('RGB', (224, 224), color = 'red')
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes = img_bytes.getvalue()

print("Posting to /api/predict...")
response = client.post("/api/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
print("STATUS CODE:", response.status_code)
print("RESPONSE BODY:", response.text)
