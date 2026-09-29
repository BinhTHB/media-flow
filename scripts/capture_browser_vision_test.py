import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path(r"C:\Users\ADMIN\.gemini\antigravity\brain\48f45660-dbcf-4460-9fca-4523c9f3659f\.tempmediaStorage")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        print("[Vision Test] 1. Truy cap /dubbing.html...")
        page.goto("http://127.0.0.1:8765/dubbing.html", wait_until="networkidle")
        time.sleep(1)
        page.screenshot(path=str(OUTPUT_DIR / "vision_01_initial.png"))
        print("[Vision Test] Chup vision_01_initial.png")

        print("[Vision Test] 2. Chon video the_other_self_30s.mp4...")
        source_item = page.locator('.source-item[data-path*="the_other_self_30s.mp4"]').first
        source_item.locator('.source-main').click()
        time.sleep(1.5)
        page.screenshot(path=str(OUTPUT_DIR / "vision_02_video_selected.png"))
        print("[Vision Test] Chup vision_02_video_selected.png")

        print("[Vision Test] 3. Bam Phan tich & tao ke hoach...")
        plan_btn = page.locator("#planButton")
        plan_btn.click()
        page.wait_for_selector(".script-item-card", timeout=30000)
        time.sleep(1.5)
        page.screenshot(path=str(OUTPUT_DIR / "vision_03_plan_loaded.png"))
        print("[Vision Test] Chup vision_03_plan_loaded.png")

        print("[Vision Test] 4. Click phat cau thoai dau tien de seek va hien subtitle overlay in-place...")
        first_card = page.locator(".script-item-card").first
        first_card.locator(".play-segment-btn").click()
        time.sleep(1.2)
        page.screenshot(path=str(OUTPUT_DIR / "vision_04_subtitle_in_place.png"))
        print("[Vision Test] Chup vision_04_subtitle_in_place.png")

        print("[Vision Test] 5. Doi subtitle layout sang Day video...")
        page.locator('.sub-opt-btn[data-layout="bottom"]').click()
        time.sleep(1.0)
        page.screenshot(path=str(OUTPUT_DIR / "vision_05_subtitle_bottom.png"))
        print("[Vision Test] Chup vision_05_subtitle_bottom.png")

        print("[Vision Test] 6. Live edit ban dich kich ban...")
        first_target = page.locator(".script-text-target").first
        first_target.fill("[BAN DICH LIVE] Xin chao MediaFlow Studio! Kiem tra dong bo phu de.")
        time.sleep(0.8)
        page.screenshot(path=str(OUTPUT_DIR / "vision_06_live_edit.png"))
        print("[Vision Test] Chup vision_06_live_edit.png")

        print("[Vision Test] 7. Chuyen sang Tab Nhan vat...")
        page.locator('[data-tab="characters"]').click()
        time.sleep(0.8)
        page.screenshot(path=str(OUTPUT_DIR / "vision_07_characters_tab.png"))
        print("[Vision Test] Chup vision_07_characters_tab.png")

        print("[Vision Test] 8. Doi giong doc AI cho Speaker 01...")
        voice_select = page.locator("#voiceSelect")
        voice_select.select_option(value="vi-VN-NamMinhNeural")
        time.sleep(0.8)
        page.screenshot(path=str(OUTPUT_DIR / "vision_08_voice_changed.png"))
        print("[Vision Test] Chup vision_08_voice_changed.png")

        print("[Vision Test] 9. Bam Them cau thoai moi...")
        page.locator('[data-tab="script"]').click()
        page.locator("#btnAddSegment").click()
        time.sleep(0.8)
        page.screenshot(path=str(OUTPUT_DIR / "vision_09_add_segment.png"))
        print("[Vision Test] Chup vision_09_add_segment.png")

        print("[Vision Test] 10. Chuyen sang /studio.html de kiem tra AI Studio TTS...")
        page.goto("http://127.0.0.1:8765/studio.html", wait_until="networkidle")
        time.sleep(1)
        page.locator("#ttsScript").fill("Chao mung ban den voi MediaFlow AI Studio. Giong doc chuyen doi thu nghiem muot ma.")
        page.locator("#ttsVoiceSelect").select_option(value="vi-VN-HoaiMyNeural")
        page.locator("#playTtsBtn").click()
        page.wait_for_selector("#audioPlayerWrapper:not([hidden])", timeout=20000)
        time.sleep(1)
        page.screenshot(path=str(OUTPUT_DIR / "vision_10_studio_tts.png"))
        print("[Vision Test] Chup vision_10_studio_tts.png")

        print("[Vision Test] 11. Chuyen sang Tab Xu ly Video tren AI Studio...")
        page.locator('.nav-tab[data-tool="video"]').click()
        time.sleep(1)
        page.screenshot(path=str(OUTPUT_DIR / "vision_11_studio_video.png"))
        print("[Vision Test] Chup vision_11_studio_video.png")

        browser.close()
        print("[Vision Test] Hoan thanh toan bo quy trinh chup anh man hinh!")

if __name__ == "__main__":
    run()
