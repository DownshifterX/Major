import urllib.request, urllib.parse, json

def fetch(url):
    r = urllib.request.urlopen(url)
    return json.loads(r.read())

BASE = 'http://127.0.0.1:8000'

# Check what IDs /historical-cases returns vs /api/search
print("=== HISTORICAL CASES — ID format ===")
hc = fetch(f'{BASE}/historical-cases')
hc_list = hc if isinstance(hc, list) else hc.get('cases', [])
print(f"Total: {len(hc_list)}")
if hc_list:
    first = hc_list[0]
    print(f"Keys: {list(first.keys())}")
    print(f"ID field: id={first.get('id')} | historical_case_id={first.get('historical_case_id')}")
    print(f"Name: {first.get('case_name')}")

print()

# Try fetching case by its 'id' field from historical-cases
if hc_list:
    case_id = hc_list[0].get('id') or hc_list[0].get('historical_case_id')
    print(f"=== CASE DETAIL via /cases/{case_id} ===")
    try:
        c = fetch(f'{BASE}/cases/{case_id}')
        if isinstance(c, dict):
            print(f"Keys: {list(c.keys())}")
            print(f"Title: {c.get('case_name', c.get('title'))}")
            print(f"Court: {c.get('court')}")
            text = c.get('text', c.get('judgment_text', ''))
            print(f"Text: {str(text)[:200]}...")
    except Exception as e:
        print(f"Error: {e}")

print()

# Check search response ID
print("=== SEARCH ID format ===")
q = urllib.parse.urlencode({'q': 'land acquisition', 'top_k': 3})
raw = fetch(f'{BASE}/api/search?{q}')
results = raw if isinstance(raw, list) else raw.get('results', [])
if results:
    print(f"Search result keys: {list(results[0].keys())}")
    search_id = results[0].get('historical_case_id', results[0].get('id'))
    print(f"Search result ID: {search_id}")
    print(f"Search result case_name: {results[0].get('case_name')}")
    
    # Try case detail with search ID
    print(f"\n=== CASE DETAIL via /cases/{search_id} ===")
    try:
        c2 = fetch(f'{BASE}/cases/{search_id}')
        print(f"Keys: {list(c2.keys())}")
        print(f"Title: {c2.get('case_name', c2.get('title'))}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Try /historical-cases/{id}
    print(f"\n=== HISTORICAL CASE DETAIL via /historical-cases/{search_id} ===")
    try:
        c3 = fetch(f'{BASE}/historical-cases/{search_id}')
        print(f"Keys: {list(c3.keys())}")
        print(f"Title: {c3.get('case_name', c3.get('title'))}")
    except Exception as e:
        print(f"Error: {e}")

print()

# Check citation graph response structure
print("=== CITATION GRAPH STRUCTURE ===")
if hc_list:
    cid = hc_list[0].get('id') or hc_list[0].get('historical_case_id')
    try:
        cg = fetch(f'{BASE}/cases/{cid}/citation-graph')
        print(f"Type: {type(cg).__name__}, Keys: {list(cg.keys()) if isinstance(cg, dict) else 'list'}")
        if isinstance(cg, dict):
            nodes = cg.get('nodes', [])
            links = cg.get('links', cg.get('edges', []))
            metrics = cg.get('metrics', {})
            print(f"Nodes count: {len(nodes)}")
            print(f"Links count: {len(links)}")
            print(f"Metrics: {metrics}")
            if nodes:
                print(f"Node sample: {nodes[0]}")
            if links:
                print(f"Link sample: {links[0]}")
    except Exception as e:
        print(f"Error: {e}")

print("\nDONE")
