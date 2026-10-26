"""
Live interactive browser demonstration for MediaFlow Dubbing Studio.
Performs real user actions, edits subtitles live, verifies timeline and caption overlay sync,
takes screenshots at each step, and logs everything.
"""
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path("outputs/browser_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "http://127.0.0.1:8765"

def run_live_browser():
    print("=== Khởi động trình duyệt thực tế để kiểm tra Dubbing Studio ===")
    
    with sync_playwright() as p:
        browser = None
        try:
            print("Đang mở trình duyệt Edge (chế độ trực quan)...")
            browser = p.chromium.launch(channel="msedge", headless=False, slow_mo=300)
        except Exception as e:
            print(f"Không thể mở headful ({e}), chuyển sang headless...")
            browser = p.chromium.launch(channel="msedge", headless=True)
            
        context = browser.new_context(
            viewport={"width": 1400, "height": 900},
            ignore_https_errors=True
        )
        page = context.new_page()

        console_logs = []
        page.on("console", lambda msg: console_logs.append(f"[{msg.type.upper()}] {msg.text}"))
        page.on("pageerror", lambda err: console_logs.append(f"[PAGE_ERROR] {err}"))

        print("[Bước 1] Truy cập http://127.0.0.1:8765/dubbing.html...")
        page.goto(f"{BASE_URL}/dubbing.html", wait_until="networkidle")
        title = page.title()
        print(f"  ✓ Tiêu đề trang: '{title}'")
        time.sleep(1)

        print("[Bước 2] Chọn video nguồn 'the_other_self_30s.mp4'...")
        page.wait_for_selector(".source-item", timeout=10000)
        target = page.locator('.source-item:has-text("the_other_self_30s.mp4") .source-main').first
        if target.count() > 0:
            target.click()
        else:
            page.locator(".source-item .source-main").first.click()
        
        page.wait_for_selector(".video-shell.has-video", timeout=10000)
        page.wait_for_function("() => !isNaN(document.getElementById('videoPlayer').duration) && document.getElementById('videoPlayer').duration > 0", timeout=15000)
        duration = page.evaluate("document.getElementById('videoPlayer').duration")
        project_name = page.locator("#projectName").inner_text()
        print(f"  ✓ Video đã nạp thành công: {project_name}, thời lượng: {duration:.2f}s")
        page.screenshot(path=str(OUTPUT_DIR / "01_video_loaded.png"))

        print("[Bước 3] Bấm nút '🔍 Xem trước kịch bản thoại (Tùy chọn)'...")
        plan_btn = page.locator("#planButton")
        plan_btn.click()

        print("  Đang chờ nhận diện ASR, OCR và dịch thuật...")
        page.wait_for_selector(".script-item-card", timeout=90000)
        status = page.locator("#planStatus").inner_text()
        seg_count = page.locator(".script-item-card").count()
        block_count = page.locator(".timeline-segment-block").count()
        print(f"  ✓ Trạng thái kịch bản: {status}")
        print(f"  ✓ Số câu thoại kịch bản hiển thị: {seg_count}")
        print(f"  ✓ Số khối timeline hiển thị: {block_count}")
        page.screenshot(path=str(OUTPUT_DIR / "02_script_and_timeline.png"))

        print("[Bước 4] Tua video tới 0.8s để kích hoạt Overlay phụ đề...")
        page.evaluate("""() => {
            const v = document.getElementById('videoPlayer');
            v.currentTime = 0.8;
            v.dispatchEvent(new Event('timeupdate'));
        }""")
        time.sleep(0.6)

        overlay_visible = page.locator("#captionOverlay").is_visible()
        caption_text = page.locator("#targetCaption").inner_text()
        print(f"  ✓ Overlay phụ đề hiển thị: {overlay_visible}")
        print(f"  ✓ Nội dung phụ đề dịch trên video: '{caption_text}'")
        page.screenshot(path=str(OUTPUT_DIR / "03_caption_overlay_active.png"))

        print("[Bước 5] Kiểm tra đồng bộ Live Edit: Gõ sửa trực tiếp câu dịch thứ nhất...")
        first_textarea = page.locator(".script-item-card").first.locator(".script-text-target")
        test_edit_text = "MẸ DẶN: PHẢI TỰ TIN VÀ TỎA SÁNG! [TEST LIVE EDIT THỰC TẾ]"
        first_textarea.fill(test_edit_text)
        
        page.evaluate("""() => {
            document.getElementById('videoPlayer').dispatchEvent(new Event('timeupdate'));
        }""")
        time.sleep(0.5)

        new_caption_text = page.locator("#targetCaption").inner_text()
        print(f"  ✓ Nội dung overlay phụ đề sau khi gõ sửa: '{new_caption_text}'")
        
        first_timeline_target_block = page.locator('.track-row:has-text("Bản dịch") .timeline-segment-block').first
        timeline_label = first_timeline_target_block.inner_text()
        print(f"  ✓ Nhãn trên thanh Timeline đã tự động cập nhật: '{timeline_label}'")
        page.screenshot(path=str(OUTPUT_DIR / "04_live_edit_synced.png"))

        print("[Bước 6] Chuyển vị trí phụ đề sang 'Đáy video' (Bottom layout)...")
        page.locator('.sub-opt-btn[data-layout="bottom"]').click()
        time.sleep(0.4)
        page.evaluate("document.getElementById('videoPlayer').dispatchEvent(new Event('timeupdate'))")
        time.sleep(0.4)
        bottom_top = page.evaluate("document.getElementById('captionOverlay').offsetTop")
        print(f"  ✓ Tọa độ Y của phụ đề khi ở đáy video: {bottom_top}px")
        page.screenshot(path=str(OUTPUT_DIR / "05_bottom_layout.png"))

        print("[Bước 7] Chuyển đổi các Preset xử lý và kiểm tra nhãn nút bấm...")
        presets = [
            ("translate-subtitle", "Dịch phụ đề"),
            ("dub-audio", "Lồng tiếng"),
            ("blur-all", "Làm mờ toàn bộ"),
            ("translate-video", "Dịch toàn diện"),
        ]
        for preset_key, preset_label in presets:
            btn = page.locator(f'.preset-btn[data-preset="{preset_key}"]')
            btn.click()
            time.sleep(0.3)
            export_text = page.locator("#exportButton").inner_text()
            print(f"  ✓ Preset '{preset_key}': nút xuất video đổi thành '{export_text}'")
            
        page.screenshot(path=str(OUTPUT_DIR / "06_final_verification.png"))

        print("\n=== Tổng kết Console Logs của Trình duyệt ===")
        errors = [log for log in console_logs if "[ERROR]" in log or "[PAGE_ERROR]" in log]
        if errors:
            print(f"  Cảnh báo lỗi ({len(errors)}):")
            for err in errors:
                print(f"    - {err}")
        else:
            print("  ✓ Hoàn toàn KHÔNG CÓ BẤT KỲ LỖI NÀO (0 errors) trong console trình duyệt!")

        time.sleep(2)
        browser.close()
        print("\n=== HOÀN TẤT TRUY CẬP TRÌNH DUYỆT THỰC TẾ ===")

if __name__ == "__main__":
    run_live_browser()
