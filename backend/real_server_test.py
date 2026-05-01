import requests
import time
import subprocess
import os

print("Starting node...")
proc = subprocess.Popen(["python", "backend/hospital_node.py", "--port", "8002", "--name", "TestNode"], cwd="D:\\codeShield_training")

time.sleep(10) # wait for boot

print("Creating image...")
from PIL import Image
import io
img = Image.new('RGB', (224, 224), color = 'red')
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes = img_bytes.getvalue()

print("Posting...")
res = requests.post("http://127.0.0.1:8002/api/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})

print("Status:", res.status_code)
print("Text:", res.text)

print("Terminating...")
proc.terminate()
