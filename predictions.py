#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import cv2
import pytesseract
import spacy
import re
import string
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# --------------------------
# Load spaCy NER model safely (absolute path)
# --------------------------
BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "output" / "model-best"

if not MODEL_DIR.exists():
    raise RuntimeError(
        f"spaCy model folder not found: {MODEL_DIR}\n"
        "Make sure 'output/model-best' exists next to predictions.py "
        "and contains config.cfg, meta.json, and model data."
    )

model_ner = spacy.load(str(MODEL_DIR))


# --------------------------
# Helpers
# --------------------------
def cleanText(txt):
    whitespace = string.whitespace
    punctuation = "!#$%&'()*+:;<=>?[\\]^`{|}~"
    tableWhitespace = str.maketrans('', '', whitespace)
    tablePunctuation = str.maketrans('', '', punctuation)
    text = str(txt)
    removewhitespace = text.translate(tableWhitespace)
    removepunctuation = removewhitespace.translate(tablePunctuation)
    return str(removepunctuation)


# Grouping labels
class groupgen():
    def __init__(self):
        self.id = 0
        self.text = ''

    def getgroup(self, text):
        if self.text == text:
            return self.id
        else:
            self.id += 1
            self.text = text
            return self.id


def parser(text, label):
    if text is None:
        return ""
    if label == 'PHONE':
        text = re.sub(r'\D', '', text.lower())
    elif label == 'EMAIL':
        allow_special_char = r'@_.\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char), '', text.lower())
    elif label == 'WEB':
        allow_special_char = r':/.%#\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char), '', text.lower())
    elif label in ('NAME', 'DES'):
        text = re.sub(r'[^a-z ]', '', text.lower()).title()
    elif label == 'ORG':
        text = re.sub(r'[^a-z0-9 ]', '', text.lower()).title()
    return text


# --------------------------
# Prediction function
# --------------------------
def getPredictions(image):
    """
    Input: OpenCV image (numpy array BGR)
    Output: (annotated_image (BGR), entities dict)
    """
    # Validate input image
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a valid OpenCV numpy array (BGR).")

    # Ensure 3 channels
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # OCR using pytesseract
    tessData = pytesseract.image_to_data(image)
    if not tessData or not isinstance(tessData, str):
        # nothing returned
        entities_empty = dict(NAME=[], ORG=[], DES=[], PHONE=[], EMAIL=[], WEB=[])
        return image.copy(), entities_empty

    tessList = list(map(lambda x: x.split('\t'), tessData.split('\n')))
    if len(tessList) <= 1:
        entities_empty = dict(NAME=[], ORG=[], DES=[], PHONE=[], EMAIL=[], WEB=[])
        return image.copy(), entities_empty

    df = pd.DataFrame(tessList[1:], columns=tessList[0])
    df.dropna(inplace=True)
    if 'text' not in df.columns:
        df['text'] = ''
    df['text'] = df['text'].apply(cleanText)

    # combine cleaned text
    df_clean = df.query('text != "" ')
    if df_clean.empty:
        entities_empty = dict(NAME=[], ORG=[], DES=[], PHONE=[], EMAIL=[], WEB=[])
        return image.copy(), entities_empty

    content = " ".join([w for w in df_clean['text']])

    # NER model prediction
    doc = model_ner(content)
    docjson = doc.to_json()
    doc_text = docjson.get('text', '')

    # tokens
    tokens_list = docjson.get('tokens', [])
    if not tokens_list:
        entities_empty = dict(NAME=[], ORG=[], DES=[], PHONE=[], EMAIL=[], WEB=[])
        return image.copy(), entities_empty

    datafram_tokens = pd.DataFrame(tokens_list)
    if {'start', 'end'}.issubset(datafram_tokens.columns):
        datafram_tokens['token'] = datafram_tokens[['start', 'end']].apply(
            lambda x: doc_text[x[0]:x[1]], axis=1
        )
    else:
        datafram_tokens['token'] = ''

    ents_list = docjson.get('ents', [])
    right_table = pd.DataFrame(ents_list) if ents_list else pd.DataFrame(columns=['start', 'label'])
    if 'start' in right_table.columns and 'label' in right_table.columns:
        datafram_tokens = pd.merge(datafram_tokens, right_table[['start', 'label']], how='left', on='start')
    datafram_tokens.fillna('O', inplace=True)

    # join labels with OCR dataframe
    df_clean = df_clean.reset_index(drop=True)
    df_clean['end'] = df_clean['text'].apply(lambda x: len(x) + 1).cumsum() - 1
    df_clean['start'] = df_clean[['text', 'end']].apply(lambda x: x[1] - len(x[0]), axis=1)

    try:
        dataframe_info = pd.merge(df_clean, datafram_tokens[['start', 'token', 'label']], how='inner', on='start')
    except Exception:
        dataframe_info = pd.DataFrame(columns=['left', 'top', 'width', 'height', 'text', 'start', 'end', 'token', 'label'])

    if dataframe_info.empty:
        entities_empty = dict(NAME=[], ORG=[], DES=[], PHONE=[], EMAIL=[], WEB=[])
        return image.copy(), entities_empty

    # entities with bounding box
    bb_df = dataframe_info.query("label != 'O' ").copy()
    if bb_df.empty:
        entities_empty = dict(NAME=[], ORG=[], DES=[], PHONE=[], EMAIL=[], WEB=[])
        return image.copy(), entities_empty

    # cleanup label prefix "B-" / "I-" if present
    bb_df['label'] = bb_df['label'].apply(lambda x: x[2:] if isinstance(x, str) and len(x) > 2 else x)
    grp_gen = groupgen()
    bb_df['group'] = bb_df['label'].apply(grp_gen.getgroup)

    # ensure geometry columns are ints
    for c in ['left', 'top', 'width', 'height']:
        bb_df[c] = pd.to_numeric(bb_df[c], errors='coerce').fillna(0).astype(int)
    bb_df['right'] = bb_df['left'] + bb_df['width']
    bb_df['bottom'] = bb_df['top'] + bb_df['height']

    # aggregate grouped entities
    col_group = ['left', 'top', 'right', 'bottom', 'label', 'token', 'group']
    group_tag_img = bb_df[col_group].groupby(by='group')
    img_tagging = group_tag_img.agg({
        'left': min,
        'right': max,
        'top': min,
        'bottom': max,
        'label': lambda x: np.unique(np.array(list(x))),
        'token': lambda x: " ".join(x)
    })

    # draw bounding boxes + labels
    img_bb = image.copy()
    for row in img_tagging.itertuples(index=False):
        l, r, t, b, label_arr, token = row

        # label may be array -> choose first
        if isinstance(label_arr, (list, tuple, np.ndarray)):
            lbl = label_arr[0] if len(label_arr) > 0 else ""
        else:
            lbl = str(label_arr)

        # clamp positions
        l, r, t, b = int(l), int(r), int(t), int(b)

        # draw bounding box
        cv2.rectangle(img_bb, (l, t), (r, b), (0, 255, 0), 2)

        # small label style for the caption (NAME/ORG/PHONE/EMAIL/DES/WEB)
        label_set = {"NAME", "ORG", "DES", "PHONE", "EMAIL", "WEB"}
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Use small font for labels
        font_scale_label = 0.6
        thickness_label = 1

        # Compute label text size
        (text_w, text_h), baseline = cv2.getTextSize(lbl, font, font_scale_label, thickness_label)
        pad = 6

        # position label above the box if space, otherwise inside
        text_x = max(0, l)
        text_y = t - 6
        if text_y - text_h - pad < 0:
            text_y = t + text_h + pad

        rect_x1 = max(0, text_x - pad)
        rect_y1 = max(0, text_y - text_h - pad)
        rect_x2 = min(img_bb.shape[1], text_x + text_w + pad)
        rect_y2 = min(img_bb.shape[0], text_y + baseline + pad)

        # draw filled rectangle for label background (small)
        cv2.rectangle(img_bb, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        # draw label text (white)
        cv2.putText(img_bb, lbl, (text_x, text_y), font, font_scale_label, (255, 255, 255), thickness_label, cv2.LINE_AA)

    # collect entities
    info_array = dataframe_info[['token', 'label']].values
    entities = dict(NAME=[], ORG=[], DES=[], PHONE=[], EMAIL=[], WEB=[])
    previous = 'O'

    for token, label in info_array:
        if not isinstance(label, str) or len(label) < 2:
            continue
        bio_tag = label[0]
        label_tag = label[2:]

        text = parser(token, label_tag)

        if bio_tag in ('B', 'I'):
            if previous != label_tag:
                entities.setdefault(label_tag, []).append(text)
            else:
                if bio_tag == "B":
                    entities.setdefault(label_tag, []).append(text)
                else:
                    if label_tag in ("NAME", "ORG", "DES"):
                        if entities.get(label_tag):
                            entities[label_tag][-1] = entities[label_tag][-1] + " " + text
                        else:
                            entities[label_tag] = [text]
                    else:
                        if entities.get(label_tag):
                            entities[label_tag][-1] = entities[label_tag][-1] + text
                        else:
                            entities[label_tag] = [text]

        previous = label_tag

    # ensure all keys exist
    for k in ['NAME', 'ORG', 'DES', 'PHONE', 'EMAIL', 'WEB']:
        entities.setdefault(k, [])

    return img_bb, entities
