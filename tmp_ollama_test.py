import urllib.request, json
url = 'http://localhost:11434/api/generate'
payload = {
    'model': 'qwen3.5:4b',
    'prompt': 'INSTRUCCIONES DEL SISTEMA:\nResponde en español muy simple y claro.\n\nUSUARIO:\n¿Qué es comercio ambulatorio?',
    'stream': False,
    'think': False,
    'options': {'temperature': 0.2, 'num_predict': 50},
}
req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=240) as resp:
        print('HTTP', resp.status)
        print(resp.read().decode('utf-8', errors='replace'))
except Exception as exc:
    import traceback
    traceback.print_exc()
