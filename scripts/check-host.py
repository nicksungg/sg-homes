"""Check real valuation HTTP calls; recover local failures and log public outages."""
import datetime as dt
import json
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATUS = ROOT / 'data/logs/hosting-health.json'
CONFIG = Path.home() / '.config/sg-homes/hosting.json'
QUERY = urllib.parse.urlencode(dict(id='406|ANG MO KIO AVE 10', type='2 ROOM', storey=11, sqm=44))


def check(base):
    with urllib.request.urlopen(base + '/api/estimate?' + QUERY, timeout=15) as response:
        data = json.load(response)
    assert 0 < data['lo'] <= data['est'] <= data['hi'], 'Invalid valuation response'
    return data['est']


state = {'checked_at': dt.datetime.now(dt.timezone.utc).isoformat()}
try:
    state['local_estimate'] = check('http://127.0.0.1:8767')
    state['local'] = 'healthy'
except Exception as error:
    state['local_error'] = str(error)
    subprocess.run(['systemctl', '--user', 'restart', 'sg-homes.service'], check=True)
    time.sleep(3)
    try:
        state['local_estimate'] = check('http://127.0.0.1:8767')
        state['local'] = 'recovered'
    except Exception as retry_error:
        state['local'] = 'failed'
        state['recovery_error'] = str(retry_error)

base = json.loads(CONFIG.read_text()).get('public_url') if CONFIG.exists() else None
if base:
    try:
        state['public_estimate'] = check(base.rstrip('/'))
        state['public'] = 'healthy'
    except Exception as error:
        state['public'] = 'unreachable'
        state['public_error'] = str(error)
else:
    state['public'] = 'not-configured'
STATUS.parent.mkdir(parents=True, exist_ok=True)
STATUS.write_text(json.dumps(state, indent=2) + '\n')
print(json.dumps(state), flush=True)
if state['local'] == 'failed' or state['public'] == 'unreachable':
    raise SystemExit(1)
