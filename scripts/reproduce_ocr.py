import os
import sys
import paddle
import paddleocr

print(f"Python version: {sys.version}")
print(f"PaddlePaddle version: {paddle.__version__}")
print(f"PaddleOCR version: {paddleocr.__version__}")

# Choose a test frame that exists
test_image = "output/job_demo_123/frames/frame_0000.jpg"
if not os.path.exists(test_image) or os.path.getsize(test_image) < 100:
    # Try finding any jpg in validation frames
    import glob
    jpgs = [f for f in glob.glob("output/**/*.jpg", recursive=True) if os.path.getsize(f) > 100]
    if jpgs:
        test_image = jpgs[0]
    else:
        print("Error: No test image found.")
        sys.exit(1)

print(f"Using test image: {test_image}")

try:
    print("Initializing PaddleOCR with engine_config to disable new IR...")
    ocr = paddleocr.PaddleOCR(
        lang="ch",
        device="cpu",
        enable_mkldnn=False,
        engine_config={"paddle_static": {"enable_new_ir": False}}
    )
    print("PaddleOCR initialized successfully.")

    print("Running OCR inference...")
    result = ocr.predict(test_image)
    print("OCR inference succeeded!")
    print(f"Result type: {type(result)}")
    print(f"Result content length: {len(result)}")
    for line in result:
        print(line)
except Exception as e:
    print(f"OCR inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
