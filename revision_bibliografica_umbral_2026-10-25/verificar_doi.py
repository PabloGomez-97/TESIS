from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,json,csv,time
R=Path(__file__).resolve().parent
DOIS=['10.1002/wcc.147','10.1029/2005JD006290','10.1038/srep04364','10.1038/s41586-024-07887-y','10.1002/ael2.70037','10.1007/s41748-024-00526-6','10.1088/1748-9326/ab7d05','10.1016/j.jhydrol.2022.127645','10.1175/JHM-D-15-0100.1','10.3354/cr031035','10.5194/hess-25-3493-2021','10.1175/JCLI-D-11-00498.1','10.1175/JCLI-D-21-0590.1','10.5194/nhess-21-1051-2021']
def get(doi):
 try:
  req=urllib.request.Request('https://api.crossref.org/works/'+doi,headers={'User-Agent':'TesisClimateLiteratureReview/1.0'})
  with urllib.request.urlopen(req,timeout=45) as res:m=json.load(res)['message']
  return dict(doi_solicitado=doi,doi=m['DOI'],titulo=m.get('title',[''])[0],revista=m.get('container-title',[''])[0],autores=[(' '.join([a.get('given',''),a.get('family','')])).strip() for a in m.get('author',[])],publicado=m.get('published',{}).get('date-parts'),publicado_online=m.get('published-online',{}).get('date-parts'),volumen=m.get('volume'),numero=m.get('issue'),paginas=m.get('page'),articulo=m.get('article-number'),url=m.get('URL'),links=m.get('link',[]),estado='verificado')
 except Exception as e:return dict(doi_solicitado=doi,estado='error',error=str(e))
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(get,DOIS))
 (R/'metadatos_crossref.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
 for d in records:print(d['doi_solicitado'],d['estado'],d.get('titulo'),d.get('autores'),d.get('publicado'))
