from pathlib import Path
p = Path("pr")
print("cwd:", Path.cwd())
print("resolved:", p.resolve())
print("exists:", p.exists())
if p.exists():
    files = sorted(p.rglob("*"))
    print("Total files found:", len(files))
    for f in files[:50]:
        print(f)
