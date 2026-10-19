"""Extract split lists + transcriptions.txt from the head of HisClima.tar.gz (they precede the 68k images)."""
import tarfile, pathlib
RAW = pathlib.Path(__file__).resolve().parent.parent / "raw"
with tarfile.open(RAW / "HisClima.tar.gz", "r|gz") as t:
    for m in t:
        if m.name.startswith("./data/lists/") or m.name == "./data/text/transcriptions.txt":
            (RAW / m.name.split("/")[-1]).write_bytes(t.extractfile(m).read())
        if m.name.endswith("/transcriptions.txt"):
            break
print("ok")
