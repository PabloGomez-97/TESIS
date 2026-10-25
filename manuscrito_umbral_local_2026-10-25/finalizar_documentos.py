from pathlib import Path
import subprocess,sys,shutil,json
from zipfile import ZipFile
BASE=Path(__file__).resolve().parent
SKILL=Path('/Users/pablogomez/.codex/plugins/cache/openai-primary-runtime/documents/26.1022.11436/skills/documents')
def run(*args):subprocess.run([sys.executable,*map(str,args)],check=True)
main=BASE/'Paper_umbral_local_1mm.docx'
tmp=BASE/'qa/paper_sin_comentarios.docx'
run(SKILL/'scripts/comments_strip.py',main,'--out',tmp);shutil.copy2(tmp,main)
for name in ['Paper_umbral_local_1mm.docx','Material_suplementario_umbral_local.docx']:
 run(SKILL/'scripts/google_docs_title_sanitize.py',BASE/name,'--in-place')
 run(SKILL/'scripts/google_docs_title_sanitize.py',BASE/name,'--check')
 inventory='paper' if name.startswith('Paper') else 'suplemento'
 ref=json.loads((BASE/'qa'/f'{inventory}_inventario.json').read_text())['source']
 original=ZipFile(BASE/'fuentes'/Path(ref).name)
 opaque={'word/header1.xml','word/numbering.xml','word/settings.xml','word/_rels/fontTable.xml.rels','_rels/.rels'}
 with ZipFile(BASE/name) as z:items=[(i,z.read(i.filename)) for i in z.infolist()]
 with ZipFile(BASE/name,'w') as z:
  for info,content in items:
   if info.filename=='word/document.xml':content=content.replace(b'Gungsuh',b'Times New Roman')
   z.writestr(info,original.read(info.filename) if info.filename in opaque and info.filename in original.namelist() else content)
