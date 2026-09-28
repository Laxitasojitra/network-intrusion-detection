"""Download the NSL-KDD train and test files into the data/ folder."""

import urllib.request

from data_prep import DATA_DIR

BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = ["KDDTrain+.txt", "KDDTest+.txt"]

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES:
        target = DATA_DIR / name
        if target.exists():
            print(f"{name} already present")
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(BASE_URL + urllib.request.quote(name), target)
    print("Done")
