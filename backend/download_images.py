import os
import requests

os.makedirs("dataset/real", exist_ok=True)
os.makedirs("dataset/fake", exist_ok=True)

real_urls = [
    "https://i.imgur.com/8zQZ6.jpg",
    "https://i.imgur.com/5bXGQ.jpg",
    "https://i.imgur.com/2nCt3.jpg"
]

fake_urls = [
    "https://i.imgur.com/ZcLLrkY.jpg",
    "https://i.imgur.com/7kCeF.jpg",
    "https://i.imgur.com/YW7nK.jpg"
]

def download(url, folder, name):
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            with open(f"{folder}/{name}.jpg", "wb") as f:
                f.write(res.content)
            print("Downloaded:", name)
        else:
            print("Failed:", url)
    except:
        print("Error:", url)

for i, url in enumerate(real_urls):
    download(url, "dataset/real", f"real_{i}")

for i, url in enumerate(fake_urls):
    download(url, "dataset/fake", f"fake_{i}")

print("✅ Images downloaded correctly")