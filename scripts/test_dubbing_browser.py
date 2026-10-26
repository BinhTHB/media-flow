"""
Comprehensive Playwright E2E Browser Test for MediaFlow Dubbing Studio (dubbing.html)
"""
import os
import sys
import time
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8765"

def run_browser_tests():
    print("=== Starting Browser E2E Test on MediaFlow Dubbing Studio ===")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: console_errors.append(str(err)))

        print("[1/7] Navigating to Dubbing Studio...")
        page.goto(f"{BASE_URL}/dubbing.html", wait_until="networkidle")
        assert "Dubbing Studio" in page.title()
        print("  ✓ Page loaded successfully")

        print("[2/7] Testing theme toggle...")
        theme_toggle = page.locator("#themeToggle")
        assert theme_toggle.is_visible()
        theme_toggle.click()
        time.sleep(0.2)
        cur_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
        print(f"  ✓ Theme switched to: {cur_theme}")
        theme_toggle.click()
        time.sleep(0.2)

        print("[3/7] Selecting sample video...")
        page.wait_for_selector(".source-item", timeout=10000)
        source_items = page.locator(".source-item")
        count = source_items.count()
        print(f"  ✓ Found {count} source videos")
        assert count > 0

        target_item = page.locator('.source-item:has-text("the_other_self_30s.mp4") .source-main').first
        if target_item.count() > 0:
            target_item.click()
        else:
            sample_item = page.locator('.source-item:has-text("sample_10s.mp4") .source-main').first
            if sample_item.count() > 0:
                sample_item.click()
            else:
                page.locator('.source-item .source-main').first.click()
        print("  ✓ Selected video item")

        page.wait_for_selector(".video-shell.has-video", timeout=10000)
        page.wait_for_function("() => !isNaN(document.getElementById('videoPlayer').duration) && document.getElementById('videoPlayer').duration > 0", timeout=15000)
        duration = page.evaluate("document.getElementById('videoPlayer').duration")
        print(f"  ✓ Video ready, duration: {duration:.2f}s")
        assert duration > 0

        print("[4/7] Generating Dubbing Plan...")
        plan_btn = page.locator("#planButton")
        assert plan_btn.is_enabled()
        plan_btn.click()

        page.wait_for_selector(".script-item-card", timeout=90000)
        status_text = page.locator("#planStatus").inner_text()
        print(f"  ✓ Plan status: {status_text}")

        segments = page.locator(".script-item-card")
        seg_count = segments.count()
        print(f"  ✓ Rendered {seg_count} script segment cards")
        assert seg_count > 0

        timeline_blocks = page.locator(".timeline-segment-block")
        block_count = timeline_blocks.count()
        print(f"  ✓ Rendered {block_count} timeline blocks")
        assert block_count > 0

        print("[5/7] Testing Subtitle Positioning & In-Place vs Bottom Layout...")
        page.evaluate("""() => {
            const v = document.getElementById('videoPlayer');
            v.currentTime = 0.5;
            v.dispatchEvent(new Event('timeupdate'));
        }""")
        time.sleep(0.5)

        overlay = page.locator("#captionOverlay")
        assert overlay.is_visible()
        print("  ✓ Caption overlay is visible at timestamp 0.5s")

        inplace_top = page.evaluate("document.getElementById('captionOverlay').offsetTop")
        inplace_left = page.evaluate("document.getElementById('captionOverlay').offsetLeft")
        inplace_width = page.evaluate("document.getElementById('captionOverlay').offsetWidth")
        inplace_height = page.evaluate("document.getElementById('captionOverlay').offsetHeight")
        inplace_classes = page.evaluate("document.getElementById('captionOverlay').className")
        print(f"  ✓ In-place overlay: top={inplace_top}px, left={inplace_left}px, width={inplace_width}px, height={inplace_height}px, class='{inplace_classes}'")

        print("  Testing switch to Bottom layout...")
        btn_bottom = page.locator('.sub-opt-btn[data-layout="bottom"]')
        btn_bottom.click()
        time.sleep(0.3)

        page.evaluate("document.getElementById('videoPlayer').dispatchEvent(new Event('timeupdate'))")
        time.sleep(0.2)

        bottom_top = page.evaluate("document.getElementById('captionOverlay').offsetTop")
        bottom_left = page.evaluate("document.getElementById('captionOverlay').offsetLeft")
        bottom_classes = page.evaluate("document.getElementById('captionOverlay').className")
        print(f"  ✓ Bottom overlay: top={bottom_top}px, left={bottom_left}px, class='{bottom_classes}'")

        assert bottom_top != inplace_top or bottom_left != inplace_left, f"Bottom layout position should differ from in-place (bottom={bottom_top}, inplace={inplace_top})"
        print(f"  ✓ Bottom layout shifted overlay position (delta top: {bottom_top - inplace_top}px)")

        print("  Testing switch back to In-place layout...")
        btn_inplace = page.locator('.sub-opt-btn[data-layout="temporal"], .sub-opt-btn[data-layout="in-place"]').first
        btn_inplace.click()
        time.sleep(0.3)

        page.evaluate("document.getElementById('videoPlayer').dispatchEvent(new Event('timeupdate'))")
        time.sleep(0.2)

        restored_top = page.evaluate("document.getElementById('captionOverlay').offsetTop")
        assert abs(restored_top - inplace_top) < 5, f"Restored top ({restored_top}) should match initial in-place top ({inplace_top})"
        print("  ✓ In-place layout restored exact coordinates")

        print("[6/7] Testing Hide/Show Source Subtitle Toggle...")
        check_hide = page.locator("#checkHideSourceSub")
        is_source_hidden = page.evaluate("""() => {
            const span = document.getElementById('sourceCaption');
            return window.getComputedStyle(span).display === 'none';
        }""")
        print(f"  ✓ Source caption hidden when checkbox checked: {is_source_hidden}")
        assert is_source_hidden

        page.evaluate("document.getElementById('checkHideSourceSub').click()")
        time.sleep(0.2)
        is_source_visible = page.evaluate("""() => {
            const span = document.getElementById('sourceCaption');
            return window.getComputedStyle(span).display !== 'none';
        }""")
        print(f"  ✓ Source caption visible when checkbox unchecked: {is_source_visible}")
        assert is_source_visible

        page.evaluate("document.getElementById('checkHideSourceSub').click()")
        time.sleep(0.2)

        print("[7/7] Testing Preset Options & Run Pipeline Button...")
        btn_trans_video = page.locator('.preset-btn[data-preset="translate-video"]')
        btn_sub = page.locator('.preset-btn[data-preset="translate-subtitle"]')
        btn_dub = page.locator('.preset-btn[data-preset="dub-audio"]')
        btn_blur = page.locator('.preset-btn[data-preset="blur-all"]')
        btn_preserve = page.locator('.preset-btn[data-preset="preserve-all"]')

        export_btn = page.locator("#exportButton")
        export_text = export_btn.inner_text()
        print(f"  ✓ Current export button (default translate-video): '{export_text}'")
        assert "translate-video" in export_text or "Dịch toàn diện" in export_text

        # Click translate-subtitle preset
        btn_sub.click()
        time.sleep(0.2)
        export_text_sub = export_btn.inner_text()
        print(f"  ✓ Export button after selecting translate-subtitle: '{export_text_sub}'")
        assert "translate-subtitle" in export_text_sub or "Dịch phụ đề" in export_text_sub

        # Click dub-audio preset
        btn_dub.click()
        time.sleep(0.2)
        export_text_dub = export_btn.inner_text()
        print(f"  ✓ Export button after selecting dub-audio: '{export_text_dub}'")
        assert "dub-audio" in export_text_dub or "Lồng tiếng" in export_text_dub

        # Click translate-video preset (which shows blur extra and subtitle layout options)
        btn_trans_video.click()
        time.sleep(0.2)
        export_text_tv = export_btn.inner_text()
        print(f"  ✓ Export button after selecting translate-video: '{export_text_tv}'")
        assert "translate-video" in export_text_tv or "Dịch toàn diện" in export_text_tv

        # Check blur extra toggle
        is_blur_checked = page.evaluate("document.getElementById('checkBlurExtra').checked")
        assert is_blur_checked
        page.evaluate("document.getElementById('checkBlurExtra').click()")
        time.sleep(0.2)
        is_blur_unchecked = page.evaluate("document.getElementById('checkBlurExtra').checked")
        assert not is_blur_unchecked
        page.evaluate("document.getElementById('checkBlurExtra').click()")
        time.sleep(0.2)
        is_blur_restored = page.evaluate("document.getElementById('checkBlurExtra').checked")
        assert is_blur_restored
        print("  ✓ Blur extra (Watermark & Logo) toggle operates smoothly")

        print("\n=== Console Errors Audit ===")
        if console_errors:
            print(f"  Found {len(console_errors)} console errors: {console_errors}")
            assert len(console_errors) == 0
        else:
            print("  ✓ ZERO console errors recorded during the entire session!")

        screenshot_path = "outputs/dubbing_studio_verified.png"
        os.makedirs("outputs", exist_ok=True)
        page.screenshot(path=screenshot_path)
        print(f"  ✓ Saved UI verification screenshot: {screenshot_path}")

        browser.close()
        print("\n=== ALL BROWSER E2E TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_browser_tests()
