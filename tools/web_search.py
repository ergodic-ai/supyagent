#!/usr/bin/env python3
"""
Web search tool - performs searches using available APIs or services
"""

import json
import re
import sys
import urllib.parse
import urllib.request


class SearchResult:
    def __init__(self, title, url, snippet):
        self.title = title
        self.url = url
        self.snippet = snippet

    def __str__(self):
        return f"{self.title}\n  {self.url}\n  {self.snippet}\n"


def search_duckduckgo(query, max_results=5):
    """
    Search using DuckDuckGo HTML interface
    Note: This is a simple implementation and may break if DDG changes their HTML
    """
    try:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0'
        }

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')

        results = []
        # Simple regex-based parsing for DuckDuckGo results
        # Looking for result blocks
        result_blocks = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            html,
            re.DOTALL
        )

        for i, (link, title_html, snippet_html) in enumerate(result_blocks[:max_results]):
            # Clean up HTML tags
            title = re.sub(r'<[^>]+>', '', title_html)
            snippet = re.sub(r'<[^>]+>', '', snippet_html)

            # Handle DDG redirect URLs
            if link.startswith('/'):
                link = 'https://duckduckgo.com' + link

            results.append(SearchResult(title.strip(), link.strip(), snippet.strip()))

        return results

    except Exception as e:
        print(f"Error searching DuckDuckGo: {e}", file=sys.stderr)
        return []


def search_google_api(query, api_key, cx, max_results=5):
    """
    Search using Google Custom Search API
    Requires API key and Custom Search Engine ID
    """
    try:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={encoded_query}&num={max_results}"

        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))

        results = []
        if 'items' in data:
            for item in data['items']:
                results.append(SearchResult(
                    item.get('title', ''),
                    item.get('link', ''),
                    item.get('snippet', '')
                ))

        return results

    except Exception as e:
        print(f"Error searching Google API: {e}", file=sys.stderr)
        return []


def main():
    if len(sys.argv) < 2:
        print("Usage: python web_search.py <query>")
        print("       python web_search.py --google <query>  (requires GOOGLE_API_KEY and GOOGLE_CX env vars)")
        sys.exit(1)

    use_google = sys.argv[1] == '--google'
    query = ' '.join(sys.argv[2:] if use_google else sys.argv[1:])

    if not query.strip():
        print("Error: Empty search query")
        sys.exit(1)

    print(f"Searching for: {query}\n")

    if use_google:
        import os
        api_key = os.environ.get('GOOGLE_API_KEY')
        cx = os.environ.get('GOOGLE_CX')

        if not api_key or not cx:
            print("Error: GOOGLE_API_KEY and GOOGLE_CX environment variables required")
            print("Get them at: https://developers.google.com/custom-search/v1/overview")
            sys.exit(1)

        results = search_google_api(query, api_key, cx)
    else:
        results = search_duckduckgo(query)

    if results:
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")
    else:
        print("No results found or search failed")


if __name__ == '__main__':
    main()
