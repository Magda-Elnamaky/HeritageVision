"""Search public museum and heritage image collections for HeritageVision & Match."""
import html
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image

HEADERS = {"User-Agent": "HeritageVisionMatch/1.1 (educational heritage search)"}
TIMEOUT = 18

# Phrase-level translation helps avoid adding every unrelated translation that
# happens to be a substring of a longer Arabic query.
ARABIC_TRANSLATIONS = {
    "مصر القديمة": "ancient Egypt",
    "فرعوني": "ancient Egyptian pharaonic",
    "فرعونية": "ancient Egyptian pharaonic",
    "فراعنة": "ancient Egyptian pharaohs",
    "مصري": "Egyptian",
    "مصرية": "Egyptian",
    "إسلامي": "Islamic art",
    "اسلامي": "Islamic art",
    "قبطي": "Coptic art",
    "قبطية": "Coptic art",
    "عثماني": "Ottoman art",
    "مخطوطة": "manuscript",
    "مخطوطات": "manuscripts",
    "سجادة": "carpet rug",
    "سجاد": "carpet rug textile",
    "فخار": "pottery",
    "خزف": "ceramics pottery",
    "تمثال": "sculpture statue",
    "تماثيل": "sculptures statues",
    "مجوهرات": "jewelry",
    "حلي": "jewelry ornaments",
    "عملة": "coin",
    "عملات": "coins",
    "نسيج تراثي": "traditional historic textile",
    "نسيج": "textile fabric",
    "تطريز": "embroidery",
    "زخرفة": "ornament decoration",
    "زخارف": "ornaments decoration",
    "مسجد": "mosque architecture",
    "كنيسة": "church architecture",
    "معبد": "ancient temple",
    "قناع": "mask",
    "أقنعة": "masks",
    "فنون شعبية": "folk art",
    "تراث شعبي": "folk art heritage",
    "حرف يدوية": "traditional handicrafts",
    "لوحة": "painting artwork",
    "لوحات": "paintings artworks",
    "فن": "artwork",
    "نقوش": "inscriptions engravings",
    "بردية": "papyrus",
    "برديات": "papyri papyrus",
    "مومياء": "mummy",
    "مومياوات": "mummies",
}


def get_search_terms(query):
    """Return the original query and a small set of useful translated queries."""
    query = re.sub(r"\s+", " ", str(query or "")).strip()
    if not query:
        return []
    terms = [query]
    translated = query
    # Longest phrases first prevents short terms from taking priority.
    for arabic, english in sorted(ARABIC_TRANSLATIONS.items(), key=lambda pair: len(pair[0]), reverse=True):
        if arabic in translated:
            translated = translated.replace(arabic, english)
    if translated != query:
        terms.append(translated)
    # Also search the most relevant translated concepts, not every matching word.
    for arabic, english in sorted(ARABIC_TRANSLATIONS.items(), key=lambda pair: len(pair[0]), reverse=True):
        if arabic in query and english not in terms:
            terms.append(english)
        if len(terms) >= 4:
            break
    return list(dict.fromkeys(term.strip() for term in terms if term.strip()))


def _clean(value):
    value = html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    return re.sub(r"\s+", " ", value).strip()


def _get_json(url, params=None, timeout=TIMEOUT):
    response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def search_commons(query, limit=12):
    results, seen = [], set()
    for term in get_search_terms(query):
        try:
            data = _get_json("https://commons.wikimedia.org/w/api.php", {
                "action": "query", "format": "json", "generator": "search",
                "gsrsearch": term, "gsrnamespace": 6, "gsrlimit": min(max(limit * 2, 10), 50),
                "prop": "imageinfo|info", "inprop": "url",
                "iiprop": "url|extmetadata", "iiurlwidth": 700,
            })
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                info_list = page.get("imageinfo", [])
                if not info_list:
                    continue
                info = info_list[0]
                image_url = info.get("thumburl") or info.get("url")
                source_url = page.get("fullurl", "")
                key = source_url or image_url
                if not image_url or key in seen:
                    continue
                seen.add(key)
                meta = info.get("extmetadata", {})
                results.append({
                    "title": _clean(page.get("title", "Untitled")),
                    "image_url": image_url, "source_url": source_url,
                    "original_url": info.get("url", image_url),
                    "description": _clean(meta.get("ImageDescription", {}).get("value", ""))[:500],
                    "license": _clean(meta.get("LicenseShortName", {}).get("value", "Check source page")),
                    "source": "Wikimedia Commons",
                })
                if len(results) >= limit:
                    return results
        except (requests.RequestException, ValueError, TypeError, KeyError):
            continue
    return results[:limit]


def search_artic(query, limit=12):
    results, seen = [], set()
    for term in get_search_terms(query):
        try:
            data = _get_json("https://api.artic.edu/api/v1/artworks/search", {
                "q": term, "limit": min(max(limit * 3, 12), 100),
                "fields": "id,title,image_id,artist_title,date_display,is_public_domain,thumbnail",
            })
            for artwork in data.get("data", []):
                image_id = artwork.get("image_id")
                artwork_id = artwork.get("id")
                if not image_id or not artwork_id or artwork_id in seen:
                    continue
                seen.add(artwork_id)
                image_base = "https://www.artic.edu/iiif/2/" + quote(str(image_id), safe="")
                results.append({
                    "title": _clean(artwork.get("title") or "Untitled"),
                    "image_url": image_base + "/full/843,/0/default.jpg",
                    "source_url": f"https://www.artic.edu/artworks/{artwork_id}",
                    "original_url": image_base + "/full/1686,/0/default.jpg",
                    "description": f"Artist: {artwork.get('artist_title') or 'Unknown'} | Date: {artwork.get('date_display') or 'Unknown'}",
                    "license": "Public Domain" if artwork.get("is_public_domain") else "Check source page",
                    "source": "Art Institute of Chicago",
                })
                if len(results) >= limit:
                    return results
        except (requests.RequestException, ValueError, TypeError, KeyError):
            continue
    return results[:limit]


def search_met(query, limit=12):
    results, seen = [], set()
    for term in get_search_terms(query):
        try:
            data = _get_json("https://collectionapi.metmuseum.org/public/collection/v1.1/search", {
                "q": term, "hasImages": "true", "limit": min(max(limit * 3, 30), 100),
            })
            object_ids = data.get("objectIDs") or []
            # A small candidate pool avoids hundreds of slow sequential API calls.
            for object_id in object_ids[:min(limit * 2, 40)]:
                if object_id in seen:
                    continue
                seen.add(object_id)
                try:
                    obj = _get_json(
                        f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}",
                        timeout=12,
                    )
                    image_url = obj.get("primaryImageSmall") or obj.get("primaryImage")
                    if not image_url:
                        continue
                    results.append({
                        "title": _clean(obj.get("title") or "Untitled"),
                        "image_url": image_url,
                        "source_url": obj.get("objectURL", ""),
                        "original_url": obj.get("primaryImage") or image_url,
                        "description": f"Artist: {obj.get('artistDisplay') or 'Unknown'} | Date: {obj.get('objectDate') or 'Unknown'} | Culture: {obj.get('culture') or 'Unknown'}",
                        "license": "Public Domain" if obj.get("isPublicDomain") else "Check source page",
                        "source": "The Metropolitan Museum of Art",
                    })
                    if len(results) >= limit:
                        return results
                except (requests.RequestException, ValueError, TypeError, KeyError):
                    continue
        except (requests.RequestException, ValueError, TypeError, KeyError):
            continue
    return results[:limit]



def search_openverse(query, limit=12):
    """Search Openverse's openly licensed image index (anonymous API access)."""
    results, seen = [], set()
    for term in get_search_terms(query)[:2]:
        try:
            data = _get_json("https://api.openverse.org/v1/images/", {
                "q": term, "page_size": min(max(limit * 2, 10), 40),
                "mature": "false",
            }, timeout=TIMEOUT)
            for item in data.get("results", []):
                image_url = item.get("thumbnail") or item.get("url")
                source_url = item.get("foreign_landing_url") or item.get("detail_url") or ""
                key = source_url or image_url
                if not image_url or key in seen:
                    continue
                seen.add(key)
                license_name = item.get("license") or "Check source page"
                license_version = item.get("license_version") or ""
                results.append({
                    "title": _clean(item.get("title") or "Untitled"),
                    "image_url": image_url,
                    "source_url": source_url,
                    "original_url": item.get("url") or image_url,
                    "description": _clean("Creator: " + str(item.get("creator") or "Unknown") + " | Provider: " + str(item.get("provider") or "Unknown"))[:500],
                    "license": (str(license_name) + (" " + str(license_version) if license_version else "")),
                    "source": "Openverse · " + str(item.get("provider") or "Openverse"),
                })
                if len(results) >= limit:
                    return results
        except (requests.RequestException, ValueError, TypeError, KeyError):
            continue
    return results[:limit]


def search_cleveland(query, limit=12):
    """Search Cleveland Museum of Art's no-key open-access collection API."""
    results, seen = [], set()
    for term in get_search_terms(query)[:2]:
        try:
            data = _get_json("https://openaccess-api.clevelandart.org/api/artworks/", {
                "q": term, "limit": min(max(limit * 2, 12), 50), "has_image": 1,
            })
            for obj in data.get("data", []):
                image = obj.get("images") or {}
                web = image.get("web") or {}
                image_url = web.get("url") or image.get("print") or image.get("full")
                object_id = obj.get("id")
                if not image_url or not object_id or object_id in seen:
                    continue
                seen.add(object_id)
                creators = obj.get("creators") or []
                creator = ", ".join(str(x.get("description") or x.get("name") or "") for x in creators[:2])
                culture = obj.get("culture") or "Unknown culture"
                results.append({
                    "title": _clean(obj.get("title") or "Untitled"),
                    "image_url": image_url,
                    "source_url": obj.get("url") or f"https://www.clevelandart.org/art/{object_id}",
                    "original_url": image.get("full") or image_url,
                    "description": _clean(f"Culture: {culture} | Creator: {creator or 'Unknown'} | Date: {obj.get('creation_date') or 'Unknown'}")[:500],
                    "license": "CC0 / Open Access when designated on the source record",
                    "source": "Cleveland Museum of Art",
                })
                if len(results) >= limit:
                    return results
        except (requests.RequestException, ValueError, TypeError, KeyError):
            continue
    return results[:limit]


def search_heritage(query, limit=12, sources=None):
    """Search selected museum APIs; returns unique results from several sources."""
    query = str(query or "").strip()
    if not query:
        return []
    limit = max(1, min(int(limit), 60))
    available = {
        "Wikimedia Commons": search_commons,
        "Art Institute of Chicago": search_artic,
        "The Metropolitan Museum of Art": search_met,
        "Openverse (صور مرخّصة من مصادر متعددة)": search_openverse,
        "Cleveland Museum of Art": search_cleveland,
    }
    selected = {name: fn for name, fn in available.items() if not sources or name in sources}
    if not selected:
        return []
    per_source = min(30, max(6, (limit * 2 + len(selected) - 1) // len(selected)))
    collected = []
    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        tasks = [executor.submit(fn, query, per_source) for fn in selected.values()]
        for task in as_completed(tasks):
            try:
                collected.extend(task.result())
            except Exception:
                continue
    unique, seen = [], set()
    for item in collected:
        key = item.get("source_url") or item.get("original_url") or item.get("image_url")
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:limit]


def _embed(embed_fn, image):
    vector = np.asarray(embed_fn(image), dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if not norm or not np.isfinite(norm):
        raise ValueError("The image model returned an invalid embedding.")
    return vector / norm


def search_heritage_by_image(image, embed_fn, limit=12, sources=None, hint=""):
    """Retrieve public collection candidates, then rank their images by visual similarity.

    This is image-to-image similarity over candidates from the chosen public APIs,
    not a reverse-image search across the entire web. embed_fn must accept a PIL image.
    """
    limit = max(1, min(int(limit), 24))
    categories = [
        "traditional craft artifact", "historic textile embroidery", "traditional pottery",
        "folk art ornament",
    ]
    if hint.strip():
        queries = [hint.strip(), *categories[:2]]
    else:
        queries = categories
    candidate_limit = max(5, min(10, (limit + len(queries) - 1) // len(queries) + 2))
    candidates, seen = [], set()
    # Use several broad categories so the upload is not dependent on one text label.
    for query in queries:
        for item in search_heritage(query, limit=candidate_limit, sources=sources):
            key = item.get("source_url") or item.get("original_url") or item.get("image_url")
            if key and key not in seen:
                seen.add(key)
                candidates.append(item)
        if len(candidates) >= 30:
            break

    query_vector = _embed(embed_fn, image)
    ranked = []
    session = requests.Session()
    session.headers.update(HEADERS)
    for item in candidates[:30]:
        try:
            response = session.get(item["image_url"], timeout=12)
            response.raise_for_status()
            if len(response.content) > 12 * 1024 * 1024:
                continue
            candidate_image = Image.open(BytesIO(response.content)).convert("RGB")
            candidate_vector = _embed(embed_fn, candidate_image)
            if candidate_vector.shape != query_vector.shape:
                continue
            similarity = float(np.dot(query_vector, candidate_vector))
            # Cosine similarity is informative for ranking; map to a bounded display scale.
            item["similarity_percent"] = round(max(0.0, min(100.0, similarity * 100.0)), 1)
            ranked.append((similarity, item))
        except Exception:
            continue
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in ranked[:limit]]