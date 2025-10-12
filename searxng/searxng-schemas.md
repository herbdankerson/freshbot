# SearXNG Schemas

## Startpage JSON payload
Querying the running instance with Startpage enforced and asking for JSON metadata:

```bash
curl 'http://127.0.0.1:8085/search?q=climate+tech&format=json&engines=startpage'
```

### High-level envelope

```json
{
  "query": "climate tech",
  "number_of_results": 0,
  "results": [ ... 10 Startpage-derived objects ... ],
  "answers": [],
  "corrections": [],
  "infoboxes": [],
  "suggestions": [],
  "unresponsive_engines": []
}
```

`number_of_results` often reports `0` even when the embedded list carries items; rely on the length of `results` for actual rows.

### Result object shape
Each `results[]` entry exposes these keys (sample values shown for the first hit of the capture above):

```json
{
  "url": "https://www.nasdaq.com/solutions/listings/resources/blogs/what-is-climate-tech",
  "title": "What is Climate Tech? | Nasdaq",
  "content": "Climate tech refers to technologies and services that enable decarbonization of the global economy.",
  "publishedDate": null,
  "engine": "startpage",
  "template": "default.html",
  "parsed_url": ["https", "www.nasdaq.com", "/solutions/listings/resources/blogs/what-is-climate-tech", "", "", ""],
  "img_src": "",
  "thumbnail": "",
  "priority": "",
  "engines": ["startpage"],
  "positions": [1],
  "score": 1.0,
  "category": "general"
}
```

Additional fields that can appear:
- `pubdate`: localized string version of `publishedDate` when SearXNG extracts a timestamp (seen on some items, e.g. MIT Technology Review result).
- `content` contains the snippet. Newlines are escaped as `\u00a0` (non-breaking spaces) instead of line breaks.
- `thumbnail` is empty because Startpage does not supply one in this configuration; other engines may fill it with a URL or base64 data URI.
- `img_src` mirrors Startpage's preview image value when provided.
- `priority` is unused by Startpage and defaults to empty string.

### Field quick reference
- `url`: canonical target link.
- `title`: headline displayed in the SERP.
- `content`: snippet text for the row; flatten whitespace before rendering.
- `publishedDate`: ISO 8601 string or `null`. Pair with `pubdate` for human-readable fallback.
- `engine`: primary upstream engine identifier (`startpage`).
- `engines`: list of all upstream engines contributing the result (single-element array here).
- `positions`: ranking indices reported by the engine (Startpage delivers `1..10`).
- `score`: SearXNG scoring weight (float).
- `category`: result category bucket (`general`).
- `template`: result template name that the web UI would use.
- `parsed_url`: six-part tuple from `urllib.parse.urlparse` for downstream parsing.
- `img_src` / `thumbnail`: preview imagery identifiers (empty unless Startpage supplies media).
- `priority`: reserved slot for future weighting; blank for Startpage.

Use these keys when mapping Startpage responses into the search-results table. No extra pagination metadata is present beyond the `results` list length; issue another call with `pageno=2` if you need more rows.
