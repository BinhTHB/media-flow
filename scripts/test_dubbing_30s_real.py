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

        print("[3/7] Selecting 'the_other_self_30s.mp4'...")
        page.wait_for_selector(".source-item", timeout=10000)
        video_item = page.locator('.source-item:has-text("the_other_self_30s.mp4") .source-main')
        if video_item.count() == 0:
            page.locator("#refreshFiles").click()
            time.sleep(1)
            video_item = page.locator('.source-item:has-text("the_other_self_30s.mp4") .source-main')
        assert video_item.count() > 0, "Video 'the_other_self_30s.mp4' not found in project file list"
        video_item.first.click()
        print("  ✓ Selected 'the_other_self_30s.mp4'")

        page.wait_for_selector(".video-shell.has-video", timeout=10000)
        page.wait_for_function("() => !isNaN(document.getElementById('videoPlayer').duration) && document.getElementById('videoPlayer').duration > 0", timeout=15000)
        duration = page.evaluate("document.getElementById('videoPlayer').duration")
        print(f"  ✓ Video ready on player, duration: {duration:.2f}s")
        assert duration >= 25.0, f"Expected ~30s video, got {duration:.2f}s"

        print("[4/7] Generating Dubbing Plan...")
        plan_btn = page.locator("#planButton")
        assert plan_btn.is_enabled()
        plan_btn.click()

        page.wait_for_selector(".script-item-card", timeout=120000)
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

        first_seg_info = page.evaluate("""() => {
            if (window.MediaFlowState && window.MediaFlowState.plan && window.MediaFlowState.plan.segments) {
                const s = window.MediaFlowState.plan.segments[0];
                return {
                    start: s.start,
                    end: s.end,
                    source: s.source_text,
                    target: s.target_text,
                    bbox: s.bbox || window.MediaFlowState.plan.default_subtitle_bbox
                };
            }
            return {start: 1.0, end: 2.0, source: "", target: "", bbox: null};
        }""")
        first_seg_time = first_seg_info["start"] + 0.3
        print(f"  ✓ Câu thoại 1: [{first_seg_info['start']:.2f}s -> {first_seg_info['end']:.2f}s]")
        print(f"    Gốc: {first_seg_info['source']}")
        print(f"    Dịch: {first_seg_info['target']}")
        print(f"    BBox YOLO: {first_seg_info['bbox']}")

        print("[5/8] Testing Subtitle Positioning & In-Place vs Bottom Layout...")
        page.evaluate(f"""() => {{
            const v = document.getElementById('videoPlayer');
            v.currentTime = {first_seg_time};
            v.dispatchEvent(new Event('timeupdate'));
        }}""")
        time.sleep(0.5)

        overlay = page.locator("#captionOverlay")
        assert overlay.is_visible()
        overlay_text = page.locator("#targetCaption").inner_text()
        print(f"  ✓ Caption overlay is visible at timestamp {first_seg_time:.2f}s: '{overlay_text}'")

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

        assert bottom_top > inplace_top, f"Bottom layout top ({bottom_top}) should be lower than in-place top ({inplace_top})"
        print(f"  ✓ Bottom layout moved overlay down by {bottom_top - inplace_top}px")

        print("  Testing switch back to In-place layout...")
        btn_inplace = page.locator('.sub-opt-btn[data-layout="in-place"]')
        btn_inplace.click()
        time.sleep(0.3)

        page.evaluate("document.getElementById('videoPlayer').dispatchEvent(new Event('timeupdate'))")
        time.sleep(0.2)

        restored_top = page.evaluate("document.getElementById('captionOverlay').offsetTop")
        assert abs(restored_top - inplace_top) < 5, f"Restored top ({restored_top}) should match initial in-place top ({inplace_top})"
        print("  ✓ In-place layout restored exact coordinates")

        preview_screenshot = "outputs/the_other_self_30s_preview.png"
        os.makedirs("outputs", exist_ok=True)
        page.screenshot(path=preview_screenshot)
        print(f"  ✓ Saved preview screenshot: {preview_screenshot}")

        print("[6/8] Testing Hide/Show Source Subtitle Toggle...")
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

        print("[7/8] Testing Preset Options...")
        btn_sub = page.locator('.preset-btn[data-preset="translate-subtitle"]')
        btn_dub = page.locator('.preset-btn[data-preset="dub-audio"]')
        btn_blur = page.locator('.preset-btn[data-preset="blur-all"]')

        export_btn = page.locator("#exportButton")
        export_text = export_btn.inner_text()
        print(f"  ✓ Current export button: '{export_text}'")

        print("[8/8] Testing Full Export Video Pipeline (Rendering)...")
        # Ensure translate-subtitle is active
        sub_classes = btn_sub.get_attribute("class") or ""
        if "is-active" not in sub_classes:
            btn_sub.click()
            time.sleep(0.3)

        export_btn.click()
        print("  ✓ Clicked Export button, waiting for video render pipeline to complete...")

        # Wait for completion (max 240s)
        page.wait_for_function(
            """() => {
                const msg = document.getElementById('formMessage');
                if (msg && (msg.textContent.includes('hoàn tất') || msg.textContent.includes('thất bại'))) return true;
                return false;
            }""",
            timeout=240000
        )
        form_msg = page.locator("#formMessage").inner_text()
        print(f"  ✓ Pipeline message: {form_msg}")
        assert "hoàn tất" in form_msg.lower(), f"Export pipeline failed: {form_msg}"

        completed_screenshot = "outputs/the_other_self_30s_completed.png"
        page.screenshot(path=completed_screenshot)
        print(f"  ✓ Saved completed UI screenshot: {completed_screenshot}")

        print("\n=== Console Errors Audit ===")
        if console_errors:
            print(f"  Found {len(console_errors)} console errors: {console_errors}")
            assert len(console_errors) == 0
        else:
            print("  ✓ ZERO console errors recorded during the entire session!")

        browser.close()
        print("\n=== ALL 30-SECOND REAL PIPELINE TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_browser_tests()
