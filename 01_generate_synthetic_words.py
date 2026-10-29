import os
import random
import glob
import cv2
import numpy as np
import pandas as pd
from PIL import Image

DATASET_CLASS_DIR = "Dataset_class"
OUTPUT_DIR = "data_crnn"
IMAGE_OUT_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_OUT_DIR, exist_ok=True)

SAMPLE_WORDS = [
    "dou", "mbojo", "kandei", "nggahi", "dana", "raha", "ina", "ama", 
    "wari", "sia", "nahu", "ita", "made", "buri", "surat", "la", "hanta",
    "bune", "habar", "mawa", "ncai", "londo", "paju", "ringi", "tente",
    "kenta", "keli", "doro", "oi", "motito", "sarume", "romba", "bade",
    "santabe", "katente", "kasipahu", "karawi", "ngaha", "nono", "taho",
    "afi", "uma", "wadu", "puu", "roo", "kacumpu", "nae", "toi", "kula",
    "bondo", "pana", "morina", "ganta", "malingi", "ica", "dua", "tolu",
    "upa", "lima", "pidu", "walu", "ciwi", "sampulu", "labo", "rasa", "ruma"
]

syllable_images = {}
for cls_name in os.listdir(DATASET_CLASS_DIR):
    cls_path = os.path.join(DATASET_CLASS_DIR, cls_name)
    if os.path.isdir(cls_path):
        imgs = glob.glob(os.path.join(cls_path, "*.jpg")) + glob.glob(os.path.join(cls_path, "*.png"))
        if imgs:
            syllable_images[cls_name.lower()] = imgs

def crop_glyph(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    mask = gray < 220
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    pad = 4
    h, w = gray.shape
    y0, y1 = max(0, y0 - pad), min(h, y1 + pad)
    x0, x1 = max(0, x0 - pad), min(w, x1 + pad)
    return img[y0:y1, x0:x1]

def pemenggal_sederhana(kata):
    suku = []
    i = 0
    k = kata.lower().replace("'", "").replace("-", "")
    n = len(k)
    while i < n:
        if i + 3 <= n and k[i:i+3] == "ngg":
            if i + 4 <= n and k[i+3] in "aiueo":
                suku.append(f"g{k[i+3]}")
                i += 4
                continue
        if i + 3 <= n and k[i:i+2] in ["mb", "nd", "mp", "nt", "nc", "nk"]:
            konsonan_dasar = k[i+1]
            suku.append(f"{konsonan_dasar}{k[i+2]}")
            i += 3
            continue
        if i + 1 < n and k[i:i+2] in syllable_images:
            suku.append(k[i:i+2])
            i += 2
        elif k[i] in syllable_images:
            suku.append(k[i])
            i += 1
        else:
            i += 1
    return suku

def generate_word_image(syl_list, target_h=40):
    cropped_glyphs = []
    for s in syl_list:
        img_path = random.choice(syllable_images[s])
        glyph = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        glyph = crop_glyph(glyph)
        
        gh, gw = glyph.shape
        new_gw = max(8, int(round(gw * target_h / gh)))
        resized = cv2.resize(glyph, (new_gw, target_h), interpolation=cv2.INTER_AREA)
        
        shift_y = random.randint(-2, 2)
        M = np.float32([[1, 0, 0], [0, 1, shift_y]])
        resized = cv2.warpAffine(resized, M, (new_gw, target_h), borderValue=255)
        
        cropped_glyphs.append(resized)
        
    gap_min, gap_max = 3, 9
    total_w = sum(g.shape[1] for g in cropped_glyphs) + gap_max * len(cropped_glyphs) + 20
    canvas = np.full((target_h + 16, total_w), 255, dtype=np.uint8)
    
    cur_x = 8
    base_y = 8
    for g in cropped_glyphs:
        gh, gw = g.shape
        canvas[base_y:base_y+gh, cur_x:cur_x+gw] = np.minimum(canvas[base_y:base_y+gh, cur_x:cur_x+gw], g)
        cur_x += gw + random.randint(gap_min, gap_max)
        
    final_img = canvas[:, :cur_x + 6]
    
    if random.random() < 0.25:
        final_img = cv2.GaussianBlur(final_img, (3, 3), 0)
    if random.random() < 0.15:
        noise = np.random.randint(0, 12, final_img.shape, dtype=np.uint8)
        final_img = cv2.subtract(final_img, noise)
        
    return final_img

def main(n_samples=3000):
    records = []
    all_keys = list(syllable_images.keys())
    
    for idx in range(n_samples):
        if random.random() < 0.6:
            word = random.choice(SAMPLE_WORDS)
            syls = pemenggal_sederhana(word)
            if not syls:
                syls = [random.choice(all_keys) for _ in range(random.randint(2, 3))]
        else:
            syls = [random.choice(all_keys) for _ in range(random.randint(2, 4))]
            word = "".join(syls)
            
        img = generate_word_image(syls, target_h=40)
        filename = f"syn_{idx:05d}.png"
        filepath = os.path.join(IMAGE_OUT_DIR, filename)
        cv2.imwrite(filepath, img)
        records.append({"filename": filename, "label": word})
            
    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[:int(0.8 * n)]
    val_df   = df.iloc[int(0.8 * n):int(0.9 * n)]
    test_df  = df.iloc[int(0.9 * n):]
    
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

if __name__ == "__main__":
    main(n_samples=3000)
