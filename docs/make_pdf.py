"""The technical document as a PDF, for the submission form: writeup.md rendered, screenshots and all.

  python docs/make_pdf.py        -> submission/The Wibblers.pdf

Needs Playwright with Chrome (pip install playwright) and an internet connection: the Markdown is rendered in the
browser by marked, loaded from jsDelivr.
"""
import json, os, tempfile

from playwright.sync_api import sync_playwright

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'submission', 'The Wibblers.pdf')
STYLE = """
body { font: 10.5pt/1.45 -apple-system, "Helvetica Neue", Arial, sans-serif; color: #1f2328; margin: 0; }
h1 { font-size: 20pt; margin: 0 0 6pt; } h2 { font-size: 15pt; margin: 18pt 0 6pt; border-bottom: 1px solid #d0d7de; }
h3 { font-size: 12pt; margin: 14pt 0 4pt; } p, li { margin: 4pt 0; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 9pt; } th, td { border: 1px solid #d0d7de; padding: 3pt 5pt; }
th { background: #f6f8fa; } tr { break-inside: avoid; }
code { font: 9pt Menlo, monospace; background: #f6f8fa; padding: 0 2pt; } pre { background: #f6f8fa; padding: 6pt; }
pre code { font-size: 8pt; } img { max-width: 100%; break-inside: avoid; } hr { border: 0; border-top: 1px solid #d0d7de; }
a { color: #0969da; text-decoration: none; }
"""


def main():
    md = open(os.path.join(REPO, 'writeup.md')).read()
    html = f"""<!doctype html><html><head><meta charset="utf-8"><base href="file://{REPO}/">
<style>{STYLE}</style><script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script></head>
<body><main id="doc"></main><script>document.getElementById('doc').innerHTML = marked.parse({json.dumps(md)});</script>
</body></html>"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        page_path = os.path.join(tmp, 'writeup.html')
        open(page_path, 'w').write(html)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='chrome')
            page = browser.new_page()
            page.goto('file://' + page_path, wait_until='networkidle')
            page.wait_for_function("document.querySelectorAll('#doc h2').length > 0")
            page.wait_for_function("[...document.images].every(i => i.complete)")
            page.pdf(path=OUT, format='A4', print_background=True,
                     margin=dict(top='16mm', bottom='16mm', left='14mm', right='14mm'))
            browser.close()
    print(f'-> {os.path.relpath(OUT, REPO)}')


if __name__ == '__main__':
    main()
