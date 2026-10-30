"""Windows-compatible equivalent of fetch_data.sh, using identical raw sources."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    meld, dd = ROOT / "meld", ROOT / "dailydialog"
    meld.mkdir(parents=True, exist_ok=True)
    dd.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        url = f"https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/{split}_sent_emo.csv"
        with urlopen(url, timeout=120) as response:
            (meld / f"{split}_sent_emo.csv").write_bytes(response.read())
    url = "https://raw.githubusercontent.com/declare-lab/conv-emotion/master/DialogueRNN/dailydialog.zip"
    with urlopen(url, timeout=120) as response:
        archive = ZipFile(BytesIO(response.read()))
    for split in ("train", "valid", "test"):
        (dd / f"{split}.json").write_bytes(archive.read(f"dailydialog/{split}.json"))
    print("Downloaded shared MELD and DailyDialog sources to", ROOT)


if __name__ == "__main__":
    main()
