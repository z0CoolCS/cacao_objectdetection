import requests
import time
url = ""

resp = requests.post(url, files={'file': open('cacao1.jpg', 'rb')}, timeout=30)
end_time = time.time()

print(resp.text)
print(end_time - start_time)
