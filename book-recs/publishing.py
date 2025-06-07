import argparse
import collections
import glob
import json
import os
import pickle
from urllib.parse import urlparse

import requests
import torch
import tqdm
from PIL import Image

from scraping import jsonl

DOWNLOAD_CACHE_DIR = "data/cache/thumbnails"
PUBLISHED_IMAGE_DIR = "site/images"


def download_image(out_basename, url) -> str | None:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0",
        "X-Requested-With": "XMLHttpRequest",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if not content_type:
        raise ValueError("Error: No Content-Type header found")

    if "jpeg" in content_type or "jpg" in content_type:
        extension = ".jpg"
    elif "png" in content_type:
        extension = ".png"
    elif "gif" in content_type:
        extension = ".gif"
    elif "webp" in content_type:
        extension = ".webp"
    elif "bmp" in content_type:
        extension = ".bmp"
    elif "svg" in content_type:
        extension = ".svg"
    else:
        raise ValueError("Unknown thumbnail content type: ", content_type)

    filepath = f"{out_basename}{extension}"
    with open(filepath, "wb") as f:
        f.write(response.content)
    return filepath


def get_image_name(basename) -> str:
    images = glob.glob(basename + "*")
    if len(images) == 0:
        return ""
    elif len(images) > 1:
        return ValueError(
            f"Found multiple paths matching basename: {basename}. Found: {images}"
        )
    return images[0]


def convert_image(image_path, converted_basename) -> str:
    out_path = converted_basename + ".webp"
    with Image.open(image_path) as img:
        goal_width = 120
        width, height = img.size

        scale = goal_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img.save(out_path, "WebP", quality=90, optimize=True)
    return out_path


def cache_convert_and_publish_thumbnail(isbn: str, thumbnail_url: str) -> None:
    os.makedirs(DOWNLOAD_CACHE_DIR, exist_ok=True)
    os.makedirs(PUBLISHED_IMAGE_DIR, exist_ok=True)

    downloaded_basename = DOWNLOAD_CACHE_DIR + f"/{isbn}"
    downloaded_path = get_image_name(downloaded_basename)
    if not downloaded_path:
        downloaded_path = download_image(downloaded_basename, thumbnail_url)

    converted_basename = PUBLISHED_IMAGE_DIR + f"/{isbn}"
    converted_path = get_image_name(converted_basename)
    if not converted_path:
        converted_path = convert_image(downloaded_path, converted_basename)


def get_published_thumbnail(isbn: str) -> str:
    published_basename = PUBLISHED_IMAGE_DIR + f"/{isbn}"
    path = get_image_name(published_basename)
    if not path:
        raise ValueError(
            f"Couldn't find published image with basename: {published_basename}"
        )
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_file", type=str, help="Path to the full data file for the dataset."
    )
    parser.add_argument(
        "features_file",
        type=str,
        help=(
            "Path to the features file to load predictions from. This should be a pickle "
            "file containing a dict of Features. Predictions will be read from the 'pred' "
            "feature."
        ),
    )
    args = parser.parse_args()

    data = jsonl.read_jsonl(args.data_file)
    with open(args.features_file, "rb") as f:
        features = pickle.load(f)

    books_list = [b for b in data if "autotranslation" not in b]
    books = {b["isbn"]: b for b in books_list}
    print(features.keys())
    for book_data, pred in zip(features["book_data"], features["pred"]):
        isbn = book_data["isbn"]
        if isbn not in books:
            continue

        if isinstance(pred, torch.Tensor):
            pred = pred.item()
        books[isbn]["pred"] = pred

    books = [b for b in books.values() if "pred" in b]

    failures = collections.defaultdict(int)
    for b in tqdm.tqdm(books, desc="Converting images"):
        if not "thumbnail_url" in b:
            print("Missing thumbnail: ", b)
            continue

        domain = urlparse(b["thumbnail_url"]).netloc
        try:
            if failures[domain] > 3:
                continue
            cache_convert_and_publish_thumbnail(b["isbn"], b["thumbnail_url"])
        except requests.exceptions.HTTPError as e:
            failures[domain] += 1
            print(e)

    out_books = []
    for b in books:
        try:
            thumbnail = get_published_thumbnail(b["isbn"])
            thumbnail = thumbnail.removeprefix("site/")
        except ValueError as e:
            thumbnail = ""

        out = {
            "title": b["title"],
            "author": b["author"],
            "thumbnail": thumbnail,
            "difficulty": b["pred"],
            "isbn": b["isbn"],
            "lang": b["lang"][:2],
        }
        out_books.append(out)
    with open("site/books.json", "w") as f:
        json.dump(out_books, f)
