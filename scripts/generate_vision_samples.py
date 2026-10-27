"""Generate clear, photorealistic or stylized silhouette demo images for Vision AI studio."""
from PIL import Image, ImageDraw
import os

os.makedirs("backend/vision_samples", exist_ok=True)

W, H = 500, 500
BG_COLOR = (15, 23, 42)      # Deep slate #0f172a
FLOOR_COLOR = (30, 41, 59)   # Slate #1e293b
FIGURE_COLOR = (248, 250, 252) # Slate-50 bright silhouette
ACCENT_COLOR = (59, 130, 246)  # Accent blue

def base_canvas():
    img = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(img)
    # Draw subtle room perspective floor line
    draw.line([(0, 440), (W, 440)], fill=FLOOR_COLOR, width=3)
    return img, draw

# 1. STANDING
img, draw = base_canvas()
# Head
draw.ellipse([235, 70, 265, 105], fill=FIGURE_COLOR)
# Neck
draw.rectangle([245, 105, 255, 120], fill=FIGURE_COLOR)
# Torso
draw.rounded_rectangle([230, 120, 270, 250], radius=8, fill=FIGURE_COLOR)
# Arms at sides
draw.line([(230, 130), (220, 245)], fill=FIGURE_COLOR, width=10)
draw.line([(270, 130), (280, 245)], fill=FIGURE_COLOR, width=10)
# Legs straight down together
draw.line([(240, 250), (240, 435)], fill=FIGURE_COLOR, width=14)
draw.line([(260, 250), (260, 435)], fill=FIGURE_COLOR, width=14)
img.save("backend/vision_samples/demo_standing.jpg", quality=95)

# 2. WALKING
img, draw = base_canvas()
# Head
draw.ellipse([235, 70, 265, 105], fill=FIGURE_COLOR)
# Torso leaning slightly forward
draw.rounded_rectangle([230, 120, 270, 245], radius=8, fill=FIGURE_COLOR)
# Swinging arms
draw.line([(232, 135), (200, 210), (180, 240)], fill=FIGURE_COLOR, width=10)
draw.line([(268, 135), (300, 210), (320, 235)], fill=FIGURE_COLOR, width=10)
# Split stride legs
draw.line([(242, 245), (210, 340), (175, 435)], fill=FIGURE_COLOR, width=14) # Front stride
draw.line([(258, 245), (290, 340), (325, 435)], fill=FIGURE_COLOR, width=14) # Rear stride
img.save("backend/vision_samples/demo_walking.jpg", quality=95)

# 3. SITTING
img, draw = base_canvas()
# Simple chair silhouette
draw.rectangle([180, 280, 200, 435], fill=FLOOR_COLOR)
draw.rectangle([190, 300, 300, 315], fill=FLOOR_COLOR)
# Head
draw.ellipse([215, 130, 245, 165], fill=FIGURE_COLOR)
# Neck
draw.rectangle([226, 163, 234, 175], fill=FIGURE_COLOR)
# Torso upright
draw.rounded_rectangle([210, 170, 250, 295], radius=8, fill=FIGURE_COLOR)
# Arms resting on lap
draw.line([(230, 185), (260, 250), (285, 290)], fill=FIGURE_COLOR, width=10)
# Thighs extending horizontally forward
draw.line([(235, 295), (320, 295)], fill=FIGURE_COLOR, width=14)
# Lower legs bent 90 degrees down to floor
draw.line([(320, 295), (320, 435)], fill=FIGURE_COLOR, width=14)
img.save("backend/vision_samples/demo_sitting.jpg", quality=95)

# 4. LAYING (Flat horizontal posture)
img, draw = base_canvas()
# Head on left
draw.ellipse([70, 360, 105, 395], fill=FIGURE_COLOR)
# Horizontal Torso
draw.rounded_rectangle([110, 365, 250, 395], radius=6, fill=FIGURE_COLOR)
# Arms extended
draw.line([(120, 370), (240, 370)], fill=FIGURE_COLOR, width=8)
# Legs extended straight horizontally
draw.line([(250, 375), (420, 375)], fill=FIGURE_COLOR, width=12)
img.save("backend/vision_samples/demo_laying.jpg", quality=95)

# 5. FALL (Crumpled fallen posture on floor)
img, draw = base_canvas()
# Floor level impact
draw.ellipse([110, 390, 140, 420], fill=FIGURE_COLOR) # Head near floor
# Torso slumped diagonally against ground
draw.line([(140, 405), (220, 425)], fill=FIGURE_COLOR, width=22)
# Limbs sprawled / crumpled
draw.line([(160, 410), (130, 445)], fill=FIGURE_COLOR, width=10) # Arm sprawled
draw.line([(180, 415), (220, 450)], fill=FIGURE_COLOR, width=10) # Arm
draw.line([(220, 425), (270, 395), (320, 440)], fill=FIGURE_COLOR, width=14) # Bent knee crumpled
draw.line([(220, 425), (290, 435), (345, 445)], fill=FIGURE_COLOR, width=12) # Leg flat
img.save("backend/vision_samples/demo_fall.jpg", quality=95)

print("Demo samples successfully generated in backend/vision_samples/")
