from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path('output/review/gemini_direct_video_g5/sample2/g59/g510_forensic')
W, H = 720, 1280

old_baseline = [743.0, 311.0, 826.0, 689.0]
gemini_cover  = [757.0, 308.0, 796.0, 693.0]

def _px(box):
    ymin, xmin, ymax, xmax = box
    return (round(xmin * W / 1000), round(ymin * H / 1000),
            round(xmax * W / 1000), round(ymax * H / 1000))

def annotate(src_name, dest_name, boxes, colors):
    img = Image.open(OUT / src_name).convert('RGB')
    draw = ImageDraw.Draw(img)
    for box, color in zip(boxes, colors):
        draw.rectangle(_px(box), outline=color, width=4)
    img.save(OUT / dest_name)

# G. source frame + old V4 bbox (red)
annotate('source_B_mid.png', 'source_B_mid_annotated_G.png', [old_baseline], [(255, 0, 0)])

# H. source frame + Gemini recommended_cover_bbox (green)
annotate('source_B_mid.png', 'source_B_mid_annotated_H.png', [gemini_cover], [(0, 255, 0)])

# I. rendered frame + recommended_cover_bbox (green)
annotate('rendered_B_mid.png', 'rendered_B_mid_annotated_I.png', [gemini_cover], [(0, 255, 0)])

print('Annotated successfully')
