from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import os
from ultralytics import YOLO

# Create FastAPI app
app = FastAPI()

# Allow CORS (frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained YOLOv12 model
model = YOLO(r"C:\Users\KEERTHAN\Desktop\finalpro\best.pt")  # Replace with your custom weights path

# Directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(result_image: str = None):
    result_html = ""
    if result_image:
        result_html = f"""
        <div class="result-section">
            <h3>Detection Result:</h3>
            <img src="{result_image}" class="result-image"/>
        </div>
        """

    return f"""
    <html>
    <head>
        <title>Camouflage Detection</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Poppins', sans-serif;
            }}
            body {{
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                background: linear-gradient(135deg, #2c3e50, #4ca1af);
                background-size: 400% 400%;
                animation: gradientBG 15s ease infinite;
                padding: 20px;
            }}
            @keyframes gradientBG {{
                0% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}
            .container {{
                background: white;
                padding: 40px 50px;
                border-radius: 20px;
                box-shadow: 0 20px 50px rgba(0,0,0,0.2);
                text-align: center;
                max-width: 650px;
                width: 100%;
                transition: all 0.3s ease-in-out;
            }}
            h2 {{
                margin-bottom: 25px;
                color: #2c3e50;
                font-weight: 600;
            }}
            input[type="file"] {{
                padding: 15px;
                border: 3px dashed #4ca1af;
                border-radius: 12px;
                width: 100%;
                margin-bottom: 25px;
                background: #f0f8ff;
                cursor: pointer;
                transition: 0.3s ease;
            }}
            input[type="file"]:hover {{
                background: #e6f7ff;
            }}
            input[type="submit"] {{
                background-color: #4ca1af;
                color: white;
                border: none;
                padding: 14px 32px;
                font-size: 16px;
                font-weight: 500;
                border-radius: 10px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }}
            input[type="submit"]:hover {{
                background-color: #3a7c8d;
            }}
            .preview {{
                margin-top: 20px;
                max-width: 100%;
                max-height: 300px;
                display: none;
                border-radius: 12px;
                border: 2px solid #4ca1af;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            }}
            .result-section {{
                margin-top: 40px;
                animation: fadeIn 1s ease-in-out;
            }}
            .result-image {{
                max-width: 100%;
                border: 3px solid #2ecc71;
                border-radius: 12px;
                margin-top: 20px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            }}
            #loading {{
                display: none;
                margin-top: 20px;
                font-weight: bold;
                color: #2ecc71;
                animation: pulse 1.5s infinite;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            @keyframes pulse {{
                0% {{ opacity: 0.6; }}
                50% {{ opacity: 1; }}
                100% {{ opacity: 0.6; }}
            }}
            @media (max-width: 680px) {{
                .container {{
                    padding: 30px 20px;
                }}
                input[type="submit"] {{
                    width: 100%;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Camouflage Detection</h2>
            <form action="/predict/" enctype="multipart/form-data" method="post" onsubmit="return showLoading()">
                <input type="file" name="file" id="fileInput" accept="image/*" onchange="previewImage(event)" required>
                <img id="preview" class="preview"/>
                <input type="submit" value="Upload & Detect">
            </form>
            <div id="loading">🔍 Processing...</div>
            {result_html}
        </div>

        <script>
            function previewImage(event) {{
                const preview = document.getElementById('preview');
                preview.src = URL.createObjectURL(event.target.files[0]);
                preview.style.display = 'block';
            }}

            function showLoading() {{
                document.getElementById('loading').style.display = 'block';
                return true;
            }}
        </script>
    </body>
    </html>
    """



# Predict route
@app.post("/predict/", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    # Save uploaded image
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Read image using OpenCV
    image = cv2.imread(input_path)

    # Run YOLOv12 inference
    results = model.predict(image, conf=0.02)

    # Draw bounding boxes for "camouflage"
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls].lower()
            if label == "camouflage":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save result image as PNG
    output_filename = f"result_{os.path.splitext(file.filename)[0]}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(output_path, image)

    # Return updated home page with the result image
    return await home(result_image=f"/outputs/{output_filename}")

# Serve output images
@app.get("/outputs/{image_name}")
async def get_output_image(image_name: str):
    return FileResponse(os.path.join(OUTPUT_DIR, image_name))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
