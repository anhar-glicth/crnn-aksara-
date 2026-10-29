import os
import glob
import cv2
import numpy as np
from PIL import Image, ImageOps

class BimaAksaraRenderer:
    def __init__(self, dataset_dir="Dataset_class"):
        self.dataset_dir = dataset_dir
        self.glyph_cache = {}
        self._load_reference_glyphs()
        
    def _crop_tight(self, gray_img):
        mask = gray_img < 220
        coords = np.argwhere(mask)
        if coords.size == 0:
            return gray_img
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        pad = 6
        h, w = gray_img.shape
        return gray_img[max(0, y0-pad):min(h, y1+pad), max(0, x0-pad):min(w, x1+pad)]

    def _load_reference_glyphs(self):
        if not os.path.exists(self.dataset_dir):
            return

        for cls_name in os.listdir(self.dataset_dir):
            cls_path = os.path.join(self.dataset_dir, cls_name)
            if os.path.isdir(cls_path):
                files = glob.glob(os.path.join(cls_path, "*.jpg")) + glob.glob(os.path.join(cls_path, "*.png"))
                if files:
                    img = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        self.glyph_cache[cls_name.lower()] = self._crop_tight(img)

    def render_syllables_to_image(self, syllables, target_height=72, gap=10, theme="parchment", **kwargs):
        if not syllables:
            return Image.fromarray(np.full((target_height, 100), 255, dtype=np.uint8))
            
        rendered_pieces = []
        for syl in syllables:
            syl_key = syl.lower()
            if syl_key in self.glyph_cache:
                glyph = self.glyph_cache[syl_key]
                gh, gw = glyph.shape
                new_w = max(14, int(round(gw * target_height / gh)))
                resized = cv2.resize(glyph, (new_w, target_height), interpolation=cv2.INTER_LANCZOS4)
                rendered_pieces.append(resized)
                
        if not rendered_pieces:
            return Image.fromarray(np.full((target_height, 100), 255, dtype=np.uint8))
            
        pad_x = 24
        pad_y = 16
        total_w = sum(p.shape[1] for p in rendered_pieces) + (len(rendered_pieces) - 1) * gap + (pad_x * 2)
        total_h = target_height + (pad_y * 2)
        
        canvas_gray = np.full((total_h, total_w), 255, dtype=np.uint8)
        
        cur_x = pad_x
        for p in rendered_pieces:
            ph, pw = p.shape
            canvas_gray[pad_y:pad_y+ph, cur_x:cur_x+pw] = np.minimum(canvas_gray[pad_y:pad_y+ph, cur_x:cur_x+pw], p)
            cur_x += pw + gap
            
        if theme == "parchment":
            bg_color = np.array([218, 238, 246], dtype=np.float32)
            ink_color = np.array([24, 28, 48], dtype=np.float32)
            
            alpha = canvas_gray.astype(np.float32) / 255.0
            alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
            
            out_bgr = (ink_color * (1.0 - alpha_3d) + bg_color * alpha_3d).astype(np.uint8)
            cv2.rectangle(out_bgr, (4, 4), (total_w - 5, total_h - 5), (150, 180, 200), 2)
            cv2.rectangle(out_bgr, (8, 8), (total_w - 9, total_h - 9), (180, 205, 220), 1)
            
            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(out_rgb)
            
        elif theme == "dark":
            bg_color = np.array([26, 18, 14], dtype=np.float32)
            ink_color = np.array([230, 245, 255], dtype=np.float32)
            
            alpha = canvas_gray.astype(np.float32) / 255.0
            alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
            
            out_bgr = (ink_color * (1.0 - alpha_3d) + bg_color * alpha_3d).astype(np.uint8)
            cv2.rectangle(out_bgr, (2, 2), (total_w - 3, total_h - 3), (70, 50, 30), 1)
            
            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(out_rgb)
            
        elif theme == "transparent":
            inv = 255 - canvas_gray
            rgba = np.zeros((total_h, total_w, 4), dtype=np.uint8)
            rgba[:, :, 0] = 30
            rgba[:, :, 1] = 30
            rgba[:, :, 2] = 30
            rgba[:, :, 3] = inv
            return Image.fromarray(rgba, mode="RGBA")
            
        else:
            return Image.fromarray(canvas_gray)

if __name__ == "__main__":
    renderer = BimaAksaraRenderer()
    sample = ["do", "u", "bo", "jo"]
    img_parch = renderer.render_syllables_to_image(sample, theme="parchment")
    img_parch.save("sample_parchment.png")
