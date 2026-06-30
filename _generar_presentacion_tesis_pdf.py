"""
Genera PDF de presentacion con todos los puntos de la tesis, figuras y explicaciones.
Salida: presentacion_tesis/Presentacion_Tesis_DrySpells.pdf
"""
import warnings
warnings.filterwarnings('ignore')

import shutil
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from matplotlib.backends.backend_pdf import PdfPages
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / 'presentacion_tesis'
FIG_DIR = OUT_DIR / 'figuras'
PDF_PATH = OUT_DIR / 'Presentacion_Tesis_DrySpells.pdf'

CHILE_EXTENT = [-76, -65, -55, -17]
FONT_PATH = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf' / 'DejaVuSans.ttf'
FONT_BOLD = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf' / 'DejaVuSans-Bold.ttf'


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_chile_geom():
    reader = shpreader.Reader(
        shpreader.natural_earth('10m', 'cultural', 'admin_0_countries')
    )
    geoms = [r.geometry for r in reader.records()
             if r.attributes.get('NAME') == 'Chile' or r.attributes.get('ADMIN') == 'Chile']
    return unary_union(geoms)


def chile_mask_aladin(lat2d, lon2d, geom):
    prep_g = prep(geom)
    return np.fromiter(
        (prep_g.contains(Point(float(x), float(y))) or geom.touches(Point(float(x), float(y)))
         for y, x in zip(lat2d.ravel(), lon2d.ravel())),
        dtype=bool, count=lat2d.size,
    ).reshape(lat2d.shape)


def save_map_delta(diff, mask, title, cbar_label, path, cmap='RdBu'):
    vals = diff.where(mask).values.ravel()
    vals = vals[np.isfinite(vals)]
    vmax = max(float(np.nanpercentile(np.abs(vals), 98)), 0.01) if vals.size else 1
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.set_extent(CHILE_EXTENT)
    im = diff.where(mask).plot.pcolormesh(
        ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=-vmax, vmax=vmax, add_colorbar=False,
    )
    fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.85, label=cbar_label)
    ax.set_title(title, fontweight='bold', fontsize=12)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_figures():
    print('Generando / copiando figuras...')
    geom = load_chile_geom()

    # --- Copiar figuras existentes ---
    copies = [
        (ROOT / 'outputs_pregunta5_aladin' / 'rr_curves_aladin.png', 'fig_rr_curves.png'),
        (ROOT / '_pregunta7_outputs' / 'tau_local_map.png', 'fig_tau_local.png'),
        (ROOT / '_pregunta9_outputs' / 'pdf_Loa_i_global.png', 'fig_p9_loa_i_global.png'),
        (ROOT / '_pregunta9_outputs' / 'pdf_Maipo_i_global.png', 'fig_p9_maipo_i_global.png'),
        (ROOT / '_pregunta9_outputs' / 'pdf_Maule_i_global.png', 'fig_p9_maule_i_global.png'),
        (ROOT / '_pregunta9_outputs' / 'pdf_Biobio_i_global.png', 'fig_p9_biobio_i_global.png'),
        (ROOT / '_pregunta9_outputs' / 'pdf_Loa_ii_mismo.png', 'fig_p9_loa_ii_mismo.png'),
        (ROOT / '_pregunta9_outputs' / 'pdf_Loa_iii_local.png', 'fig_p9_loa_iii_local.png'),
        (ROOT / '_pregunta9_outputs' / 'pdf_Maipo_ii_mismo.png', 'fig_p9_maipo_ii_mismo.png'),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, FIG_DIR / dst)
            print(f'  copiado: {dst}')

    # --- Pregunta 3: mapas diferencia wet days e intensidad ---
    try:
        print('  generando: pregunta3 mapas delta...')
        pr_ala = xr.open_mfdataset(
            sorted(str(p) for p in (ROOT / 'pr1').glob('pr_CHP12_*_historical_*.nc')),
            use_cftime=True, chunks={'time': 365},
        )['pr'].sel(time=slice('1980-01-01', '2014-12-31')) * 86400.0
        pr_cr2 = xr.open_mfdataset(
            sorted(str(p) for p in (ROOT / 'pr').glob('CR2MET_pr_v2.5_day_*.nc')),
            chunks={'time': 365},
        )['pr'].sel(time=slice('1980-01-01', '2014-12-31')).interp(
            lat=pr_ala['lat'], lon=pr_ala['lon'], method='linear',
        )
        lat2d, lon2d = pr_ala['lat'].values, pr_ala['lon'].values
        mask = xr.DataArray(
            chile_mask_aladin(lat2d, lon2d, geom),
            coords={'y': pr_ala['y'], 'x': pr_ala['x'], 'lat': pr_ala['lat'], 'lon': pr_ala['lon']},
            dims=['y', 'x'],
        )
        wet_cr2 = (pr_cr2 >= 1.0).mean('time') * 100
        wet_ala = (pr_ala >= 1.0).mean('time') * 100
        diff_wet = (wet_ala - wet_cr2).assign_coords(lat=pr_ala['lat'], lon=pr_ala['lon'])
        save_map_delta(
            diff_wet, mask,
            'Delta fraccion wet days (ALADIN - CR2MET)\n1980-2014 | tau = 1 mm/dia',
            'Delta wet days (pp)', FIG_DIR / 'fig_p3_delta_wetdays.png',
        )
        mw_cr2 = pr_cr2.where(pr_cr2 >= 1.0).mean('time')
        mw_ala = pr_ala.where(pr_ala >= 1.0).mean('time')
        diff_mw = (mw_ala - mw_cr2).assign_coords(lat=pr_ala['lat'], lon=pr_ala['lon'])
        save_map_delta(
            diff_mw, mask,
            'Delta precip. media solo wet days (ALADIN - CR2MET)\n1980-2014 | tau = 1 mm/dia',
            'Delta (mm/dia)', FIG_DIR / 'fig_p3_delta_intensity.png',
        )
    except Exception as e:
        print(f'  aviso pregunta3: {e}')

    # --- Mapas delta CR2MET P1/P2 ---
    try:
        print('  generando: mapas_delta_pr...')
        ds = xr.open_mfdataset(str(ROOT / 'pr' / 'CR2MET_pr_v2.5_day_*.nc'), combine='by_coords')
        pr = ds['pr']
        p1 = pr.sel(time=slice('1980-01-01', '2000-12-31'))
        p2 = pr.sel(time=slice('2001-01-01', '2021-12-31'))
        wf1 = (p1 >= 1.0).mean('time') * 100
        wf2 = (p2 >= 1.0).mean('time') * 100
        delta = wf2 - wf1
        lat = ds['lat'].values
        lon = ds['lon'].values
        LON, LAT = np.meshgrid(lon, lat)
        prep_g = prep(geom)
        m = np.array([prep_g.contains(Point(x, y)) for y, x in zip(LAT.ravel(), LON.ravel())]).reshape(LAT.shape)
        mask_cr2 = xr.DataArray(m, coords={'lat': ds['lat'], 'lon': ds['lon']}, dims=['lat', 'lon'])
        delta_da = delta.assign_coords(lat=ds['lat'], lon=ds['lon'])
        save_map_delta(
            delta_da, mask_cr2,
            'Delta fraccion wet days CR2MET (P2 - P1)\n2001-2021 minus 1980-2000',
            'Delta (pp)', FIG_DIR / 'fig_delta_cr2met_wetdays.png', cmap='RdBu',
        )
    except Exception as e:
        print(f'  aviso mapas_delta: {e}')

    # --- Pregunta 6: curva F(tau) ---
    try:
        print('  generando: pregunta6 F(tau)...')
        if 'pr_ala' not in dir():
            pr_ala = xr.open_mfdataset(
                sorted(str(p) for p in (ROOT / 'pr1').glob('pr_CHP12_*_historical_*.nc')),
                use_cftime=True, chunks={'time': 365},
            )['pr'].sel(time=slice('1980-01-01', '2014-12-31')) * 86400.0
            pr_cr2 = xr.open_mfdataset(
                sorted(str(p) for p in (ROOT / 'pr').glob('CR2MET_pr_v2.5_day_*.nc')),
                chunks={'time': 365},
            )['pr'].sel(time=slice('1980-01-01', '2014-12-31')).interp(
                lat=pr_ala['lat'], lon=pr_ala['lon'], method='linear',
            )
            lat2d, lon2d = pr_ala['lat'].values, pr_ala['lon'].values
            mask = xr.DataArray(chile_mask_aladin(lat2d, lon2d, geom), dims=['y', 'x'])

        def domain_fwet(pr, tau, mask_da):
            f = (pr >= tau).mean('time') * 100
            v = f.where(mask_da).values.ravel()
            return float(np.nanmean(v[np.isfinite(v)]))

        grid = np.linspace(0, 8, 161)
        curve_ala = [domain_fwet(pr_ala, t, mask) for t in grid]
        f_cr2_1 = domain_fwet(pr_cr2, 1.0, mask)
        # bisection tau*
        lo, hi = 0.0, 15.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if domain_fwet(pr_ala, mid, mask) > f_cr2_1:
                lo = mid
            else:
                hi = mid
        tau_star = 0.5 * (lo + hi)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(grid, curve_ala, 'steelblue', lw=2.2, label='ALADIN F(tau)')
        ax.scatter([1.0], [f_cr2_1], s=90, c='firebrick', zorder=5, label=f'CR2MET @ 1 mm (F={f_cr2_1:.1f}%)')
        ax.scatter([tau_star], [f_cr2_1], s=90, facecolors='white', edgecolors='firebrick',
                   linewidth=2, marker='s', zorder=5, label=f'ALADIN tau*={tau_star:.2f} mm')
        ax.axhline(f_cr2_1, color='firebrick', ls=':', alpha=0.6)
        ax.set_xlabel('Umbral tau (mm/dia)')
        ax.set_ylabel('Fraccion integrada wet days Chile (%)')
        ax.set_title('Calibracion tau* ALADIN vs CR2MET (1980-2014)', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIG_DIR / 'fig_p6_ftau.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'  aviso pregunta6: {e}')

    # --- Tabla resumen numerico (matplotlib) ---
    print('  generando: tablas resumen...')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    rows = [
        ['Punto', 'Metrica', 'Hallazgo clave'],
        ['1', 'Wet days (tau=1mm)', 'CR2MET 20.9% vs ALADIN 30.7% (+9.7 pp); ~97% celdas sig.'],
        ['2', 'Intensidad wet days', 'CR2MET 8.66 vs ALADIN 13.55 mm/d (+4.97 mm/d)'],
        ['3', 'CR2MET P2-P1', 'Wet days 37.1% -> 36.2%; sequedizacion moderada'],
        ['4', 'Calibracion tau*', 'CR2MET 1 mm -> ALADIN tau*=5.285 mm'],
        ['5', 'Duracion media spells', 'CR2MET 42.5 d vs ALADIN 35.1 d (Forma A, tau*)'],
        ['5', 't99 spells', 'CR2MET 221.8 d vs ALADIN 178.2 d'],
        ['8', 'Tendencia ALADIN hist', 'Pendiente -0.027 d/decada; 0.035% celdas sig.'],
        ['Futuro', 'RR @ 20d Coquimbo', '1.27 (15.4% -> 19.6%); IC95 no cruza 1'],
        ['Futuro', 'RR @ 20d Araucania', '1.84 (0.74% -> 1.37%); aumento sig.'],
    ]
    table = ax.table(cellText=rows, loc='center', cellLoc='left', colWidths=[0.08, 0.28, 0.64])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for j in range(3):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    ax.set_title('Resumen numerico de hallazgos', fontweight='bold', fontsize=14, pad=20)
    fig.savefig(FIG_DIR / 'fig_tabla_resumen.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    generate_p9_table_figure()


def load_p9_resumen_i_global():
    """Carga resumen pregunta 9 (criterio i). Valida CSV; si esta obsoleto usa valores corregidos."""
    csv_path = ROOT / '_pregunta9_outputs' / 'resumen_pregunta9.csv'
    fallback = pd.DataFrame([
        ['i_global', 'Loa', 'CR2MET (hist)', 55015, 55.3, 23, 208, 646, 1419],
        ['i_global', 'Loa', 'ALADIN (hist)', 64625, 49.5, 21, 189, 598, 1433],
        ['i_global', 'Loa', 'ALADIN (futuro)', 60949, 52.4, 22, 210, 699, 1853],
        ['i_global', 'Maipo', 'CR2MET (hist)', 137557, 8.4, 4, 27, 61, 301],
        ['i_global', 'Maipo', 'ALADIN (hist)', 150867, 7.8, 4, 26, 62, 344],
        ['i_global', 'Maipo', 'ALADIN (futuro)', 142213, 8.1, 4, 29, 74, 399],
        ['i_global', 'Maule', 'CR2MET (hist)', 159576, 9.2, 4, 30, 69, 334],
        ['i_global', 'Maule', 'ALADIN (hist)', 179317, 8.2, 4, 28, 65, 349],
        ['i_global', 'Maule', 'ALADIN (futuro)', 166996, 8.8, 4, 31, 78, 419],
        ['i_global', 'Biobio', 'CR2MET (hist)', 292423, 5.7, 3, 17, 40, 127],
        ['i_global', 'Biobio', 'ALADIN (hist)', 327068, 5.3, 3, 16, 38, 137],
        ['i_global', 'Biobio', 'ALADIN (futuro)', 313058, 5.7, 3, 17, 42, 153],
    ], columns=[
        'criterio', 'cuenca', 'dataset', 'n_spells',
        'mean_dias', 'median_dias', 'p90_dias', 'p99_dias', 'max_dias',
    ])
    if not csv_path.exists():
        return fallback
    df = pd.read_csv(csv_path)
    mcol = 'mean_dias' if 'mean_dias' in df.columns else 'mean'
    sub = df[df['criterio'] == 'i_global'].copy()
    if sub.empty or float(sub[mcol].max()) > 500:
        print('  aviso: resumen_pregunta9.csv obsoleto; usando valores corregidos')
        return fallback
    rename = {
        'mean': 'mean_dias', 'median': 'median_dias',
        'p90': 'p90_dias', 'p99': 'p99_dias', 'max': 'max_dias',
    }
    return sub.rename(columns={k: v for k, v in rename.items() if k in sub.columns})


def generate_p9_table_figure():
    print('  generando: tabla pregunta 9...')
    df = load_p9_resumen_i_global()
    header = ['Cuenca', 'Dataset', 'n spells', 'Media (d)', 'Mediana', 'p90', 'p99', 'Max']
    rows = [header]
    for _, r in df.iterrows():
        rows.append([
            r['cuenca'],
            r['dataset'],
            f"{int(r['n_spells']):,}".replace(',', '.'),
            f"{r['mean_dias']:.1f}",
            f"{int(r['median_dias'])}",
            f"{int(r['p90_dias'])}",
            f"{int(r['p99_dias'])}",
            f"{int(r['max_dias'])}",
        ])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    table = ax.table(cellText=rows, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    for j in range(len(header)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows)):
        if i % 3 == 1:
            table[(i, 0)].set_facecolor('#f5f5f5')
    ax.set_title(
        'Resumen dry spells por cuenca | Criterio i (tau* global: CR2MET 1 mm, ALADIN 5.285 mm)\n'
        'Pool regional 1980-2014 (hist) vs 2040-2074 (futuro ALADIN SSP5-8.5)',
        fontweight='bold', fontsize=11, pad=16,
    )
    fig.savefig(FIG_DIR / 'fig_p9_tabla_resumen.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    df.to_csv(ROOT / '_pregunta9_outputs' / 'resumen_pregunta9_corregido.csv', index=False)


class TesisPDF(FPDF):
    W = 190  # ancho util de texto (A4 con margen 10 mm)

    def __init__(self):
        super().__init__()
        self.add_font('DV', '', str(FONT_PATH))
        self.add_font('DV', 'B', str(FONT_BOLD))
        self.set_margins(10, 10, 10)
        self.set_auto_page_break(auto=True, margin=18)

    def para(self, text, h=5.5, align='L', bold=False, size=10.5):
        """Bloque de texto; siempre vuelve al margen izquierdo (evita texto en el borde derecho)."""
        self.set_x(self.l_margin)
        self.set_font('DV', 'B' if bold else '', size)
        self.multi_cell(
            self.W, h, text, align=align,
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )

    def footer(self):
        self.set_y(-12)
        self.set_font('DV', '', 9)
        self.set_text_color(120, 120, 120)
        self.set_x(self.l_margin)
        self.cell(0, 8, f'Presentacion Tesis - Dry Spells Chile | pagina {self.page_no()}', align='C')
        self.set_text_color(0, 0, 0)

    def cover(self):
        self.add_page()
        self.ln(40)
        self.para('Evaluacion de dry spells en Chile:\nCR2MET vs ALADIN CHP12', h=12, align='C', bold=True, size=22)
        self.ln(8)
        self.para(
            'Documento de presentacion con figuras y explicaciones\n'
            'Periodo historico 1980-2014 | Futuro SSP5-8.5 2040-2074',
            h=8, align='C', size=14,
        )
        self.ln(20)
        self.para(
            'Contenido: puntos 1-5, calibracion tau*, tendencias historicas,\n'
            'proyeccion futura (risk ratios) y PDFs por cuenca.',
            h=6, align='C', size=11,
        )

    def section(self, num, title):
        self.add_page()
        self.set_text_color(30, 60, 120)
        self.para(f'Punto {num}: {title}', h=9, bold=True, size=16)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def subsection(self, title):
        self.ln(2)
        self.para(title, h=7, bold=True, size=12)
        self.ln(1)

    def body(self, text):
        self.para(text, h=5.5, size=10.5)
        self.ln(2)

    def bullet_list(self, items):
        for item in items:
            self.para(f'  - {item}', h=5.5, size=10.5)
        self.ln(2)

    def figure(self, path, caption, max_w=185):
        p = Path(path)
        if not p.exists():
            self.para(f'[Figura no disponible: {p.name}]', h=5, size=10)
            return
        self.ln(2)
        w, h = self.get_image_size(p, max_w)
        if self.get_y() + h + 25 > 280:
            self.add_page()
        x = (210 - w) / 2
        self.image(str(p), x=x, w=w)
        self.ln(2)
        self.set_text_color(80, 80, 80)
        self.para(caption, h=5, size=9)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def get_image_size(self, path, max_w):
        from PIL import Image
        with Image.open(path) as im:
            iw, ih = im.size
        ratio = ih / iw
        w = min(max_w, 185)
        return w, w * ratio


def build_pdf():
    pdf = TesisPDF()
    pdf.cover()

    # Intro
    pdf.add_page()
    pdf.para('Arco narrativo de la tesis', h=8, bold=True, size=14)
    pdf.ln(3)
    pdf.body(
        'Esta tesis evalua precipitacion y periodos secos en Chile comparando CR2MET (referencia) '
        'y ALADIN CHP12 (modelo regional). El hilo metodologico central es que comparar ambos con '
        'el mismo umbral fijo de 1 mm/dia sesga las conclusiones. La calibracion tau* (Martinez-Villalobos '
        'et al. 2022) permite comparaciones equitativas antes de analizar dry spells.'
    )
    pdf.bullet_list([
        'Puntos 1-2: diagnostico de sesgo humedo de ALADIN (frecuencia e intensidad).',
        'Punto 3: sequedizacion moderada en CR2MET entre 1980-2000 y 2001-2021.',
        'Punto 4: calibracion del umbral ALADIN tau* = 5.285 mm.',
        'Punto 5: ALADIN produce rachas mas cortas que CR2MET incluso calibrado.',
        'Punto 8 (pregunta 8): sin tendencia robusta en ALADIN historico 1980-2014.',
        'Futuro: ALADIN SSP5-8.5 proyecta mas dry spells largos en varias regiones.',
    ])
    pdf.figure(FIG_DIR / 'fig_tabla_resumen.png', 'Tabla resumen de hallazgos numericos principales.')

    # Objetivos y datos
    pdf.add_page()
    pdf.para('Objetivos y datos', h=8, bold=True, size=14)
    pdf.ln(3)
    pdf.subsection('Objetivos')
    pdf.bullet_list([
        'Caracterizar diferencias CR2MET-ALADIN en wet days e intensidad (1980-2014).',
        'Cuantificar cambios CR2MET P1 (1980-2000) vs P2 (2001-2021).',
        'Calibrar umbral tau* para comparaciones equitativas (Martinez-Villalobos 2022).',
        'Comparar climatologia de dry spells y tendencias en ALADIN historico.',
        'Evaluar cambio futuro en probabilidad de dry spells largos (RR + bootstrap).',
    ])
    pdf.subsection('Datos')
    pdf.bullet_list([
        'CR2MET v2.5: referencia observacional, grilla lat/lon 1D.',
        'ALADIN CHP12: modelo regional, grilla nativa (~3587 celdas Chile).',
        'Regrillado: CR2MET interpolado linealmente a centros ALADIN.',
        'ALADIN: kg/m2/s x 86400 -> mm/dia; CR2MET ya en mm/dia.',
        'Futuro: ALADIN SSP5-8.5, periodo 2040-2074 (35 anos).',
    ])

    # Punto 1
    pdf.section('1', 'El sesgo humedo de ALADIN (mismo tau = 1 mm)')
    pdf.body(
        'No es que ALADIN este mal, sino que comparar ambos productos usando 1 mm/dia como umbral '
        'para definir dia seco o humedo es metodologicamente injusto. Con tau = 1 mm en ambos, '
        'ALADIN registra sustancialmente mas wet days que CR2MET.'
    )
    pdf.subsection('Numeros clave (1980-2014, Chile, grilla ALADIN)')
    pdf.bullet_list([
        'Fraccion wet days: CR2MET 20.9% vs ALADIN 30.7% (+9.7 pp).',
        'ALADIN tiene ~47% mas wet days en tasa relativa.',
        '~97-100% del territorio con diferencia significativa (bootstrap por anos).',
    ])
    pdf.subsection('Definicion')
    pdf.body(
        'Wet day: dia con precipitacion diaria >= 1 mm. Dry spell: racha consecutiva de dias '
        'con pr < tau. El umbral tau es el criterio de clasificacion diaria, no la lluvia media del dia.'
    )
    pdf.figure(FIG_DIR / 'fig_p3_delta_wetdays.png',
               'Mapa delta fraccion wet days (ALADIN - CR2MET). Areas hachuradas en notebook original: significancia bootstrap.')
    pdf.subsection('Mensaje para la reunion')
    pdf.body(
        'Antes de hablar de sequia o dry spells, hay que definir que cuenta como dia humedo (wet day). '
        'Este punto justifica la calibracion tau* del punto 4.'
    )

    # Punto 2
    pdf.section('2', 'El sesgo no es solo frecuencia: tambien intensidad')
    pdf.body(
        'ALADIN no solo llueve mas dias; cuando llueve, llueve con mas intensidad. '
        'La metrica "precipitacion media solo en wet days" aísla el efecto de intensidad '
        '(ignora los dias secos y promedia solo dias con pr >= 1 mm).'
    )
    pdf.bullet_list([
        'Precip. media todos los dias: CR2MET 2.60 vs ALADIN 5.42 mm/d (+2.82).',
        'Precip. media solo wet days: CR2MET 8.66 vs ALADIN 13.55 mm/d (+4.97).',
        '98.1% de celdas con diferencia significativa en intensidad.',
    ])
    pdf.figure(FIG_DIR / 'fig_p3_delta_intensity.png',
               'Delta precipitacion media en dias lluviosos (ALADIN - CR2MET). Sesgo sistemico positivo en casi todo Chile.')

    # Punto 3
    pdf.section('3', 'CR2MET muestra sequedizacion reciente (P1 vs P2)')
    pdf.body(
        'Separar sesgo del modelo (puntos 1-2) de senal climatica en la referencia. '
        'Solo CR2MET, grilla nativa, comparando P1 (1980-2000) vs P2 (2001-2021).'
    )
    pdf.bullet_list([
        'Wet days: 37.1% -> 36.2% (delta -0.90 pp).',
        'Precip. media: 3.85 -> 3.63 mm/d (delta -0.21).',
        'Intensidad wet days: 8.84 -> 8.40 mm/d (delta -0.44).',
        'Solo ~20% de pixeles con aumento -> sequedizacion predominante en centro-sur.',
    ])
    pdf.subsection('Lectura del mapa')
    pdf.body(
        'Delta = P2 - P1. Rojo = disminucion (mas seco en P2). Azul = aumento. '
        'Predominan zonas rojas (~80% pixeles con delta negativo en wet days).'
    )
    pdf.figure(FIG_DIR / 'fig_delta_cr2met_wetdays.png',
               'Delta fraccion wet days CR2MET (2001-2021 minus 1980-2000). Sequedizacion moderada y heterogenea.')

    # Punto 4
    pdf.section('4', 'Calibracion tau* (Pregunta 6)')
    pdf.body(
        'Buscar en ALADIN el umbral tau* que iguala la fraccion integrada de wet days de CR2MET '
        'con 1 mm. Metodologia Martinez-Villalobos et al. (2022), seccion 3b. '
        'F(tau) = promedio espacial sobre Chile del % temporal de dias con pr >= tau.'
    )
    pdf.bullet_list([
        'CR2MET @ 1 mm: F = 20.94%.',
        'ALADIN @ 1 mm (sin ajuste): F = 30.68% (+9.7 pp sesgo).',
        'ALADIN tau* = 5.285 mm -> F = 20.94% (error < 0.0001 pp).',
        'Tambien: CR2MET @ 0.1 mm -> ALADIN tau* = 3.67 mm.',
    ])
    pdf.subsection('Como explicarlo oralmente')
    pdf.body(
        'Fijamos CR2MET en 1 mm y buscamos que umbral en ALADIN produce la misma frecuencia '
        'de dias humedos sobre Chile. Ese valor (~5.3 mm) se usa en criterio i (global) '
        'para preguntas 7, 8, 9 y risk ratios futuros.'
    )
    pdf.figure(FIG_DIR / 'fig_p6_ftau.png',
               'Curva F(tau) de ALADIN. Punto rojo: CR2MET @ 1 mm. Cuadrado: ALADIN tau* con misma F.')

    # Punto 5
    pdf.section('5', 'ALADIN acorta dry spells vs CR2MET (Pregunta 7)')
    pdf.body(
        'Incluso con tau* calibrado (Forma A), ALADIN subestima la persistencia de sequia. '
        'Dry spell = racha consecutiva de dias con pr < tau. Duracion media = promedio de '
        'longitud de cada racha (dias), NO el numero de rachas.'
    )
    pdf.subsection('Forma A - tau* integrado (1980-2014)')
    pdf.bullet_list([
        'Duracion media: CR2MET 42.5 d vs ALADIN 35.1 d (delta -7.9 d).',
        't99 (p99): CR2MET 221.8 d vs ALADIN 178.2 d (delta -44.7 d).',
        '73% celdas significativas; 65% con ALADIN significativamente mas corto.',
        'Sin calibracion (Forma B, 1 mm ambos): delta -27.5 d, 98% sig.',
    ])
    pdf.subsection('Tres criterios de umbral')
    pdf.bullet_list([
        'i global: CR2MET 1 mm, ALADIN 5.285 mm.',
        'ii mismo: 1 mm en ambos (control / sesgo maximo).',
        'iii local: ALADIN tau*(x,y) por pixel.',
    ])
    pdf.figure(FIG_DIR / 'fig_tau_local.png',
               'Mapa tau* local por pixel (Forma C). Calibracion espacialmente variable.')

    # Punto 8 historico
    pdf.section('8', 'Sin tendencia robusta en ALADIN historico (Pregunta 8)')
    pdf.body(
        'IMPORTANTE: Este punto NO es el futuro SSP5-8.5. Pregunta 8 analiza solo ALADIN '
        'historico 1980-2014: tendencia lineal anual de duracion de dry spells (tau* = 5.285 mm).'
    )
    pdf.bullet_list([
        'Duracion media: pendiente -0.027 dias/decada (casi cero).',
        'Solo 0.035% de celdas con tendencia significativa (p < 0.05).',
        't99: pendiente +0.001 dias/decada; 0.12% celdas significativas.',
    ])
    pdf.body(
        'Contraste con punto 3: CR2MET si muestra sequedizacion P1/P2 en precipitacion, '
        'pero ALADIN historico no detecta tendencia equivalente en persistencia de dry spells.'
    )

    # Futuro
    pdf.section('Futuro', 'ALADIN proyecta mas dry spells largos (SSP5-8.5)')
    pdf.body(
        'Pregunta 5 (notebook pregunta5.ipynb): Risk Ratio = P_futuro / P_historico para '
        'spells >= D dias. Periodos: hist 1980-2014 vs futuro 2040-2074 (35 anos c/u). '
        'tau* = 5.285 mm. Rachas con inicio marzo-noviembre. Pool regional por region administrativa.'
    )
    pdf.subsection('RR @ 20 dias (significativo)')
    pdf.bullet_list([
        'Coquimbo: RR 1.27 (15.4% -> 19.6%); IC95 [1.09, 1.48].',
        "O'Higgins: RR 1.30 (5.8% -> 7.5%); IC95 [1.04, 1.63].",
        'La Araucania: RR 1.84 (0.74% -> 1.37%); IC95 [1.16, 3.12].',
        'Los Lagos: RR 2.24 pero NO significativo (IC cruza 1).',
    ])
    pdf.body(
        'RR > 1: mayor probabilidad en el futuro. Puntos rellenos en curvas: IC95 completamente '
        'por encima de RR=1. La senal se refuerza para umbrales mas altos (cola de spells muy largos).'
    )
    pdf.figure(FIG_DIR / 'fig_rr_curves.png',
               'Curvas Risk Ratio vs umbral de duracion (4 regiones). Estilo Martinez-Villalobos & Neelin 2018.')

    # Pregunta 9
    pdf.section('9', 'PDFs de dry spells por cuenca (Pregunta 9)')
    pdf.body(
        'Cuatro cuencas hidrograficas: Loa, Maipo, Maule, Biobio. Se agrupan todos los dry spells '
        'de todos los pixeles de cada cuenca (pool regional). Criterio i: CR2MET con 1 mm/dia, '
        'ALADIN con tau* integrado 5.285 mm/dia. Tres series: CR2MET hist (1980-2014), '
        'ALADIN hist (1980-2014), ALADIN futuro SSP5-8.5 (2040-2074).'
    )
    pdf.subsection('Tabla resumen (duraciones en dias)')
    pdf.body(
        'n_spells = cantidad de eventos en el pool (no dias totales). mean_dias = duracion promedio '
        'de cada racha. En Loa (arido) las rachas son mucho mas largas que en Biobio (sur humedo).'
    )
    pdf.figure(
        FIG_DIR / 'fig_p9_tabla_resumen.png',
        'Estadisticas del pool regional por cuenca y periodo (criterio i global).',
        max_w=190,
    )
    pdf.subsection('Lectura rapida de la tabla')
    pdf.bullet_list([
        'Loa: media ~50-55 d; max hasta 1419-1853 d (spells muy largos en desierto).',
        'Maipo/Maule: media ~8-9 d; medianas ~4 d.',
        'Biobio: media ~5-6 d; spells cortos (clima mas humedo).',
        'Futuro ALADIN: en Loa max sube a 1853 d; en cuencas centrales p99 y max aumentan.',
    ])
    pdf.subsection('PDFs empiricas (escala log-log)')
    pdf.body(
        'Eje X: duracion del dry spell (dias). Eje Y: densidad de probabilidad p(t). '
        'Las tres curvas suelen ser paralelas en escala log-log (ley de potencia). '
        'CR2MET (negro) vs ALADIN hist (azul): buen acuerdo en Maipo/Maule/Biobio; '
        'en Loa ALADIN hist queda ligeramente debajo (menos probabilidad en spells largos). '
        'ALADIN futuro (rojo punteado): cola derecha puede extenderse vs historico.'
    )
    for basin in ['Loa', 'Maipo', 'Maule', 'Biobio']:
        pdf.figure(
            FIG_DIR / f'fig_p9_{basin.lower()}_i_global.png',
            f'PDF dry spells — Cuenca {basin} | Criterio i, tau* global 5.285 mm.',
        )

    # Glosario
    pdf.add_page()
    pdf.para('Glosario rapido', h=8, bold=True, size=14)
    pdf.ln(3)
    glossary = [
        ('tau / umbral', 'Corte en mm/dia para clasificar dia seco (pr < tau) vs humedo (pr >= tau).'),
        ('tau*', 'Umbral en ALADIN que iguala fwet integrado de CR2MET @ 1 mm (= 5.285 mm global).'),
        ('Wet day', 'Dia con precipitacion diaria >= umbral tau.'),
        ('Dry spell', 'Racha consecutiva de dias secos (pr < tau).'),
        ('Duracion media', 'Promedio de longitud (dias) de cada dry spell, no numero de eventos.'),
        ('t99 / p99', 'Percentil 99 de duraciones; mide spells extremadamente largos.'),
        ('fwet / F(tau)', 'Fraccion integrada de wet days sobre Chile.'),
        ('RR (Risk Ratio)', 'P_futuro / P_historico para eventos >= D dias.'),
        ('Pool regional', 'Se agrupan todos los spells de todos los pixeles de la region/cuenca.'),
    ]
    for term, defn in glossary:
        pdf.para(term, h=6, bold=True, size=10.5)
        pdf.para(defn, h=5.5, size=10)
        pdf.ln(1)

    # Cierre
    pdf.add_page()
    pdf.para('Mensaje final', h=8, bold=True, size=14)
    pdf.ln(3)
    pdf.body(
        'CR2MET y ALADIN no son intercambiables sin calibrar umbrales. La referencia CR2MET '
        'muestra sequedizacion moderada reciente. ALADIN reproduce la frecuencia de wet days '
        'con tau*, pero subestima la persistencia de dry spells vs CR2MET. El historico ALADIN '
        'no muestra tendencia robusta en 1980-2014, pero la proyeccion SSP5-8.5 sugiere mayor '
        'probabilidad de dry spells largos en regiones clave. Implicancia: comparaciones naive '
        'con 1 mm sesgan resultados; la calibracion tau* es necesaria pero no suficiente; '
        'el riesgo futuro requiere analisis regional.'
    )
    pdf.ln(5)
    pdf.para(
        'Notebooks: pregunta3, mapas_delta_pr, pregunta6, pregunta7, pregunta8, pregunta5, pregunta9.',
        h=5, size=10,
    )

    pdf.output(str(PDF_PATH))
    print(f'PDF generado: {PDF_PATH}')


if __name__ == '__main__':
    import sys
    ensure_dirs()
    if '--pdf-only' not in sys.argv:
        generate_figures()
    build_pdf()
