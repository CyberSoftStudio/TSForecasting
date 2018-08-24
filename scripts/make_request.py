import requests
import numpy as np
import math
import sys
import json
print(sys.getdefaultencoding())

signal = np.sin(np.linspace(0, 3 * math.pi, 256))

r = requests.get('http://0.0.0.0:5000/prediction', data=json.dumps({'window':list(signal)}))
print(r.text)
