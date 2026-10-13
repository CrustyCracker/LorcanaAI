import urllib.request
import json
import re
import os
import sys

def fetch_apollo_cache(url):
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
    
    idx = html.find('window.__APOLLO_CACHE__ = ')
    if idx == -1:
        return None
    start = idx + len('window.__APOLLO_CACHE__ = ')
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(html[start:])
    return obj

def scrape_guide(guide_base_url, intro_slug, out_file):
    print(f"Scraping guide from {guide_base_url}/{intro_slug}...")
    cache = fetch_apollo_cache(f"{guide_base_url}/{intro_slug}")
    if not cache:
        print("Failed to fetch initial Apollo cache.")
        return
    
    chapters = []
    for k, v in cache.items():
        if isinstance(v, dict) and v.get('__typename') == 'API_ProductGuideChapter':
            chapters.append({
                'id': v.get('id'),
                'title': v.get('title'),
                'slug': v.get('slug'),
                'position': v.get('position', 999),
                'content': v.get('content')
            })
    
    # Sort by position
    chapters.sort(key=lambda x: (x['position'] if x['position'] is not None else 999))
    print(f"Found {len(chapters)} chapters.")
    
    full_text = []
    for ch in chapters:
        title = ch['title']
        slug = ch['slug']
        pos = ch['position']
        content = ch.get('content')
        
        # If content is missing, fetch that chapter page
        if not content:
            ch_url = f"{guide_base_url}/{slug}"
            print(f"Fetching chapter {pos}: {title} ({slug})...")
            try:
                ch_cache = fetch_apollo_cache(ch_url)
                if ch_cache:
                    for ck, cv in ch_cache.items():
                        if isinstance(cv, dict) and cv.get('__typename') == 'API_ProductGuideChapter':
                            if cv.get('slug') == slug and cv.get('content'):
                                content = cv.get('content')
                                break
            except Exception as e:
                print(f"Error fetching {slug}: {e}")
        
        full_text.append(f"\n\n# Chapter {pos}: {title}\n\n")
        full_text.append(content if content else "*[No content or locked]*")
    
    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("".join(full_text))
    print(f"Saved {len(chapters)} chapters to {out_file} (length: {sum(len(t) for t in full_text)} chars).")

if __name__ == '__main__':
    # 1. Princess Aggro
    scrape_guide(
        "https://metafy.gg/guides/view/set-13-green-yellow-princess-aggro-bG0MBBHYBuk",
        "introduction-p7gMOZT9FWN",
        "data/guides/jddr_set13_princess_aggro.md"
    )
    # 2. Purple Steel Droids
    scrape_guide(
        "https://metafy.gg/guides/view/set-13-purple-steel-droids-guide-6EkVcDZiujC",
        "introduction-yBuVq6gENJD",
        "data/guides/jddr_set13_purple_steel_droids.md"
    )
