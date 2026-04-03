"""
Format references to ACS style and add missing DOIs via CrossRef API.

Input:  manuscript/drafts/references.md
Output: manuscript/drafts/references_acs.md  (formatted, for review)
        manuscript/drafts/references_report.txt (summary of changes)

ACS format:
  Author1; Author2, et al. Title. *J. Abbrev.* **Year**, *Vol* (*Issue*), Pages. https://doi.org/xxx

Usage:
    python scripts/format_references.py
"""

import re
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "manuscript" / "drafts" / "references.md"
OUTPUT_FILE = BASE_DIR / "manuscript" / "drafts" / "references_acs.md"
REPORT_FILE = BASE_DIR / "manuscript" / "drafts" / "references_report.txt"

CROSSREF_API = "https://api.crossref.org/works"
MAILTO = "chengxw@lzu.edu.cn"  # polite pool

# ============================================================
# Step 1: Parse references from markdown
# ============================================================

def clean_multiline(text: str) -> str:
    """Fix broken lines in reference text (e.g., chemical formulas split across lines)."""
    # Collapse whitespace runs (including newlines) into single space
    text = re.sub(r'\s+', ' ', text).strip()
    # Fix HTML entities
    text = text.replace('&amp;', '&')
    # Fix <scp> tags
    text = re.sub(r'</?scp>', '', text)
    return text


def parse_references(text: str) -> list[dict]:
    """Parse references.md into structured dicts."""
    refs = []
    # Split into individual reference blocks
    # Pattern: [number] followed by text until next [number] or end
    blocks = re.split(r'\n(?=\[\d+\])', text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = re.match(r'\[(\d+)\]\s*(.*)', block, re.DOTALL)
        if not m:
            continue

        num = int(m.group(1))
        raw = clean_multiline(m.group(2))

        ref = {'num': num, 'raw': raw, 'doi': None, 'doi_source': 'none'}

        # Extract existing DOI
        doi_match = re.search(r'https?://doi\.org/(10\.\S+?)(?:\s|$)', raw)
        if doi_match:
            ref['doi'] = doi_match.group(1).rstrip('.')
            ref['doi_source'] = 'original'
        else:
            # Check for bare DOI
            doi_match2 = re.search(r'(10\.\d{4,}/\S+?)(?:\s|$)', raw)
            if doi_match2:
                ref['doi'] = doi_match2.group(1).rstrip('.')
                ref['doi_source'] = 'original'

        # Extract title (between first ". " after authors and next "*")
        title_match = re.search(r'\.\s+([^*]+?)\s*\*', raw)
        if title_match:
            ref['title'] = title_match.group(1).strip().rstrip('.')
        else:
            ref['title'] = ''

        # Extract journal (between * markers)
        journal_match = re.search(r'\*([^*]+)\*', raw)
        if journal_match:
            ref['journal'] = journal_match.group(1).strip()
        else:
            ref['journal'] = ''

        # Extract year
        year_match = re.search(r'\((\d{4})\)', raw)
        if year_match:
            ref['year'] = year_match.group(1)
        else:
            ref['year'] = ''

        # Extract volume, issue, pages
        # Pattern after journal: Volume (Issue), Pages (Year)
        after_journal = raw.split('*')[-1] if '*' in raw else ''
        vol_match = re.match(r'\s*(\d+)\s*(?:\(([^)]+)\))?\s*,?\s*([\d–\-]+(?:–[\d–\-]+)?)?\s*\((\d{4})\)', after_journal)
        if vol_match:
            ref['volume'] = vol_match.group(1)
            ref['issue'] = vol_match.group(2) or ''
            ref['pages'] = vol_match.group(3) or ''
            if not ref['year']:
                ref['year'] = vol_match.group(4)
        else:
            # Try simpler patterns
            simple_vol = re.search(r'(\d+)\s*(?:\(([^)]+)\))?\s*,\s*(\S+)', after_journal)
            if simple_vol:
                ref['volume'] = simple_vol.group(1)
                ref['issue'] = simple_vol.group(2) or ''
                ref['pages'] = simple_vol.group(3).rstrip(',').rstrip('.')
            else:
                ref['volume'] = ''
                ref['issue'] = ''
                ref['pages'] = ''

        # Extract authors (everything before the title)
        authors_match = re.match(r'(.+?)\.\s+(?=[A-Z])', raw)
        if authors_match:
            ref['authors'] = authors_match.group(1).strip()
        else:
            ref['authors'] = ''

        refs.append(ref)

    return refs


# ============================================================
# Step 2: Query CrossRef API for missing DOIs
# ============================================================

def query_crossref(title: str, authors: str = '', year: str = '') -> dict | None:
    """Query CrossRef API to find DOI for a reference."""
    query = title
    if year:
        query += f" {year}"

    params = {
        'query.bibliographic': query,
        'rows': 3,
        'select': 'DOI,title,short-container-title,container-title,published-print,published-online,volume,issue,page,author',
        'mailto': MAILTO,
    }

    url = f"{CROSSREF_API}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': f'ML4Env-RefFormatter/1.0 (mailto:{MAILTO})'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  API error: {e}")
        return None

    items = data.get('message', {}).get('items', [])
    if not items:
        return None

    # Find best match by title similarity
    target = title.lower().strip()
    for item in items:
        item_titles = item.get('title', [])
        for it in item_titles:
            # Simple similarity: check if most words match
            t_words = set(re.findall(r'\w+', target))
            i_words = set(re.findall(r'\w+', it.lower()))
            if len(t_words) < 3:
                continue
            overlap = len(t_words & i_words) / max(len(t_words), 1)
            if overlap > 0.7:
                result = {
                    'doi': item.get('DOI', ''),
                    'short_journal': (item.get('short-container-title') or [''])[0],
                    'journal': (item.get('container-title') or [''])[0],
                    'volume': item.get('volume', ''),
                    'issue': item.get('issue', ''),
                    'page': item.get('page', ''),
                }
                # Extract year
                pub = item.get('published-print') or item.get('published-online') or {}
                parts = pub.get('date-parts', [[]])
                if parts and parts[0]:
                    result['year'] = str(parts[0][0])
                return result

    return None


# ============================================================
# Step 3: Format to ACS style
# ============================================================

def normalize_pages(pages: str) -> str:
    """Normalize page ranges to use en-dash."""
    if not pages:
        return ''
    return pages.replace('-', '–').replace('––', '–')


def format_acs(ref: dict) -> str:
    """Format a parsed reference into ACS style."""
    parts = []

    # [num]
    parts.append(f"[{ref['num']}]")

    # Authors
    authors = ref.get('authors', '')
    if authors:
        parts.append(f" {authors}.")

    # Title
    title = ref.get('title', '')
    if title:
        if not title.endswith('.'):
            title += '.'
        parts.append(f" {title}")

    # Journal (italicized)
    journal = ref.get('journal', '')
    if journal:
        parts.append(f" *{journal}*")

    # Year (bold)
    year = ref.get('year', '')
    if year:
        parts.append(f" **{year}**,")

    # Volume (italic)
    volume = ref.get('volume', '')
    if volume:
        parts.append(f" *{volume}*")

    # Issue (italic, in parens)
    issue = ref.get('issue', '')
    if issue:
        parts.append(f" (*{issue}*),")
    elif volume:
        parts.append(",")

    # Pages
    pages = normalize_pages(ref.get('pages', ''))
    if pages:
        parts.append(f" {pages}.")
    elif volume:
        # No pages - just close with period
        # Remove trailing comma
        last = parts[-1]
        if last.endswith(','):
            parts[-1] = last[:-1] + '.'
        else:
            parts.append(".")

    # DOI
    doi = ref.get('doi', '')
    if doi:
        parts.append(f" https://doi.org/{doi}")

    return ''.join(parts)


# ============================================================
# Main
# ============================================================

def main():
    print("Reading references...")
    text = INPUT_FILE.read_text(encoding='utf-8')
    refs = parse_references(text)
    print(f"Parsed {len(refs)} references")

    # Count DOI status
    has_doi = sum(1 for r in refs if r['doi'])
    missing_doi = sum(1 for r in refs if not r['doi'])
    print(f"  With DOI: {has_doi}")
    print(f"  Without DOI: {missing_doi}")

    # Query CrossRef for missing DOIs
    found = 0
    not_found = []
    for ref in refs:
        if ref['doi']:
            continue
        title = ref.get('title', '')
        if not title or len(title) < 10:
            not_found.append(ref['num'])
            continue

        print(f"  [{ref['num']}] Querying: {title[:60]}...")
        result = query_crossref(title, ref.get('authors', ''), ref.get('year', ''))

        if result and result.get('doi'):
            ref['doi'] = result['doi']
            ref['doi_source'] = 'crossref'
            found += 1
            print(f"    -> Found: {result['doi']}")
            # Fill in missing metadata from CrossRef
            if not ref.get('volume') and result.get('volume'):
                ref['volume'] = result['volume']
            if not ref.get('issue') and result.get('issue'):
                ref['issue'] = result['issue']
            if not ref.get('pages') and result.get('page'):
                ref['pages'] = result['page']
        else:
            not_found.append(ref['num'])
            print(f"    -> Not found")

        time.sleep(0.3)  # Rate limiting

    print(f"\nCrossRef results: {found} found, {len(not_found)} not found")
    if not_found:
        print(f"  Not found: {not_found}")

    # Format all references to ACS style
    print("\nFormatting to ACS style...")
    output_lines = ["# References\n"]
    for ref in refs:
        formatted = format_acs(ref)
        output_lines.append(formatted)
        output_lines.append("")  # blank line between refs

    OUTPUT_FILE.write_text('\n'.join(output_lines), encoding='utf-8')
    print(f"Saved: {OUTPUT_FILE}")

    # Write report
    report_lines = [
        "Reference Formatting Report",
        "=" * 40,
        f"Total references: {len(refs)}",
        f"DOIs from original: {has_doi}",
        f"DOIs from CrossRef: {found}",
        f"DOIs still missing: {len(not_found)}",
        "",
        "References without DOI:",
    ]
    for n in not_found:
        r = next(r for r in refs if r['num'] == n)
        report_lines.append(f"  [{n}] {r.get('title', 'NO TITLE')[:80]}")

    report_lines.append("")
    report_lines.append("References with DOI from CrossRef:")
    for r in refs:
        if r['doi_source'] == 'crossref':
            report_lines.append(f"  [{r['num']}] {r['doi']}")

    REPORT_FILE.write_text('\n'.join(report_lines), encoding='utf-8')
    print(f"Saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()
