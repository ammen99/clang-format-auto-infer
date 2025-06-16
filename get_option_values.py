import requests
import sys

def fetch_html_content(url):
    """Fetches HTML content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}", file=sys.stderr)
        return None

if __name__ == '__main__':
    url = "https://clang.llvm.org/docs/ClangFormatStyleOptions.html"
    html_content = fetch_html_content(url)

    if html_content:
        print(html_content)
    else:
        sys.exit(1) # Exit with an error code if fetching failed
