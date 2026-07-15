"""
plots.py — Flight Analyzer
===========================
Toutes les visualisations du projet.

Fonctions publiques :
    plot_trajectory(df, flight_id=None)          → carte Folium (HTML 2D)
    plot_trajectory_3d(df, flight_id)            → carte PyDeck (3D)
    plot_altitude_profile(df, flight_id=None)    → profil altitude Plotly
    plot_speed_vario(df, flight_id)              → vitesse + vario Plotly
    plot_dashboard(df, flight_id)                → dashboard complet un vol
    plot_comparison(df_summary)                  → comparaison multi-vols
    plot_wind_rose(df, flight_id=None)           → rose des vents
    plot_fuel_accuracy(df_summary)               → précision modèle carburant
"""

import os
import numpy as np
import pandas as pd
import folium
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pydeck as pdk
from pathlib import Path


# ─────────────────────────────────────────────
# CONSTANTES VISUELLES & FONDS DE CARTE
# ─────────────────────────────────────────────

# Couleurs par phase
PHASE_COLORS = {
    "ground":  "#8c8c8c",  # gris
    "takeoff": "#f5c518",  # jaune
    "climb":   "#f97316",  # orange
    "cruise":  "#38bdf8",  # bleu clair
    "descent": "#22c55e",  # vert
    "landing": "#a855f7",  # violet
}

PHASE_LABELS = {
    "ground":  "Sol",
    "takeoff": "Décollage",
    "climb":   "Montée",
    "cruise":  "Croisière",
    "descent": "Descente",
    "landing": "Atterrissage",
}

FLIGHT_PALETTE = [
    "#e63946", "#2196f3", "#ff9800", "#4caf50",
    "#9c27b0", "#00bcd4", "#ff5722", "#607d8b",
]

OUTPUT_DIR = Path("outputs")

# --- LISTE DES FONDS DE CARTE FIABLES ---
MAP_TILES = {
    "Carte Relief VFR (Stamen Terrain)": {
        "url": "https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}{r}.png",
        "attr": "© <a href='https://stadiamaps.com/'>Stadia Maps</a>, © <a href='https://stamen.com/'>Stamen Design</a>"
    },
    "Esri Satellite (Vue Réelle)": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "© Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP"
    },
    "OpenTopoMap (Topographique)": {
        "url": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "© OpenTopoMap, © OpenStreetMap"
    },
    "OpenStreetMap (Standard)": {
        "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": "© OpenStreetMap contributors"
    }
}


# ─────────────────────────────────────────────
# UTILITAIRES INTERNES
# ─────────────────────────────────────────────

def _ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)

def _get_flight(df: pd.DataFrame, flight_id: str) -> pd.DataFrame:
    """Filtre le DataFrame sur un flight_id. Lève une erreur claire si absent."""
    if flight_id not in df["flight_id"].values:
        available = df["flight_id"].unique().tolist()
        raise ValueError(f"flight_id '{flight_id}' introuvable. Disponibles : {available}")
    return df[df["flight_id"] == flight_id].sort_values("datetime_utc")

def _flight_color(df: pd.DataFrame, flight_id: str) -> str:
    """Retourne une couleur cohérente pour un flight_id donné."""
    ids = sorted(df["flight_id"].unique())
    idx = ids.index(flight_id) % len(FLIGHT_PALETTE)
    return FLIGHT_PALETTE[idx]

def _center(g: pd.DataFrame) -> tuple:
    """Retourne le centre géographique d'un groupe de points."""
    return g["lat"].mean(), g["lon"].mean()


# ─────────────────────────────────────────────
# 1. CARTE INTERACTIVE 2D (FOLIUM)
# ─────────────────────────────────────────────

def plot_trajectory(
    df: pd.DataFrame,
    flight_id: str = None,
    color_by: str = "phase",
    map_style: str = "Carte Relief VFR (Stamen Terrain)",
    openaip_key: str = None, 
    save: bool = True,
) -> folium.Map:
    """
    Carte interactive de la trajectoire avec choix du fond de carte.
    """
    _ensure_output_dir()

    if flight_id:
        flights = {flight_id: _get_flight(df, flight_id)}
    else:
        flights = {fid: _get_flight(df, fid) for fid in sorted(df["flight_id"].unique())}

    first = next(iter(flights.values()))
    center = _center(first)

    # Récupération du fond de carte choisi
    tile_info = MAP_TILES.get(map_style, MAP_TILES["OpenStreetMap (Standard)"])
    
    # Création de la carte de base
    m = folium.Map(
        location=center,
        zoom_start=9,
        tiles=tile_info["url"],
        attr=tile_info["attr"],
        control_scale=True
    )

    for fid, g in flights.items():
        coords = list(zip(g["lat"], g["lon"]))

        if color_by == "phase" and "phase" in g.columns:
            _add_phase_polyline(m, g, fid)
        elif color_by == "altitude":
            _add_colored_polyline(m, g, "altitude_ft", fid)
        elif color_by == "speed":
            _add_colored_polyline(m, g, "speed_kt", fid)
        else:
            color = _flight_color(df, fid)
            folium.PolyLine(
                coords, color=color, weight=3, opacity=0.85, tooltip=fid,
            ).add_to(m)

        # Marqueurs Départ/Arrivée
        folium.Marker(
            location=coords[0],
            icon=folium.Icon(color="green", icon="plane", prefix="fa"),
            popup=f"<b>Départ</b><br>{fid}<br>{g['datetime_utc'].iloc[0].strftime('%H:%M UTC')}",
        ).add_to(m)

        folium.Marker(
            location=coords[-1],
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
            popup=f"<b>Arrivée</b><br>{fid}<br>{g['datetime_utc'].iloc[-1].strftime('%H:%M UTC')}",
        ).add_to(m)

        # Popups sur les points (échantillonnage pour les performances)
        for _, row in g.iloc[::50].iterrows():
            popup_html = (
                f"<b>{fid}</b><br>"
                f"⏱ {row['datetime_utc'].strftime('%H:%M:%S')} UTC<br>"
                f"✈ Alt : {row['altitude_ft']} ft<br>"
                f"💨 Vitesse : {row['speed_kt']} kt<br>"
                f"🧭 Cap : {row['heading_deg']}°"
            )
            if "phase" in g.columns:
                popup_html += f"<br>📍 Phase : {PHASE_LABELS.get(row['phase'], row['phase'])}"

            folium.CircleMarker(
                location=(row["lat"], row["lon"]),
                radius=4,
                color=PHASE_COLORS.get(row.get("phase", "cruise"), "#38bdf8"),
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=220),
            ).add_to(m)
            
    # ... fin de la boucle for fid, g in flights.items() ...

    # ← ICI : Overlay aéronautique OpenAIP
    if openaip_key:
        folium.TileLayer(
            tiles=f"https://api.tiles.openaip.net/api/data/openaip/{{z}}/{{x}}/{{y}}.png?apiKey={openaip_key}",
            attr="© <a href='https://www.openaip.net'>OpenAIP</a>",
            name="Données aéronautiques (OpenAIP)",
            overlay=True,
            control=True,
            show=True,
            opacity=0.9,
            max_zoom=17,
            min_zoom=4,
        ).add_to(m)

    if color_by == "phase":
        _add_phase_legend(m)
    
    # ...suite du code...

    if color_by == "phase":
        _add_phase_legend(m)

    if save:
        fname = OUTPUT_DIR / f"trajectory_{'_'.join(flights.keys())}.html"
        m.save(str(fname))

    return m

def _add_phase_polyline(m: folium.Map, g: pd.DataFrame, fid: str):
    """Trace la trajectoire segment par segment, coloré par phase."""
    phases = g["phase"].values
    lats   = g["lat"].values
    lons   = g["lon"].values

    i = 0
    while i < len(g) - 1:
        phase = phases[i]
        color = PHASE_COLORS.get(phase, "#38bdf8")

        j = i + 1
        while j < len(g) and phases[j] == phase:
            j += 1

        segment_coords = list(zip(lats[i:j+1], lons[i:j+1]))
        folium.PolyLine(
            segment_coords,
            color=color,
            weight=4,
            opacity=0.9,
            tooltip=f"{fid} — {PHASE_LABELS.get(phase, phase)}",
        ).add_to(m)
        i = j

def _add_colored_polyline(m: folium.Map, g: pd.DataFrame, col: str, fid: str):
    """Trace la trajectoire colorée par gradient sur une colonne numérique."""
    values = g[col].values
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    colormap = cm.get_cmap("RdYlGn")

    lats = g["lat"].values
    lons = g["lon"].values

    for i in range(len(g) - 1):
        norm = (values[i] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        rgba = colormap(norm)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
        )
        folium.PolyLine(
            [(lats[i], lons[i]), (lats[i+1], lons[i+1])],
            color=hex_color,
            weight=4,
            opacity=0.9,
        ).add_to(m)

def _add_phase_legend(m: folium.Map):
    """Ajoute une légende HTML des phases sur la carte."""
    legend_items = "".join([
        f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
        f'<div style="width:16px;height:16px;background:{color};border-radius:3px"></div>'
        f'<span>{PHASE_LABELS[phase]}</span></div>'
        for phase, color in PHASE_COLORS.items()
    ])
    legend_html = f"""
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: white; padding: 12px 16px; border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25); font-family: Arial; font-size: 13px;
    ">
        <b style="display:block;margin-bottom:6px">Phases de vol</b>
        {legend_items}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


# ─────────────────────────────────────────────
# 2. CARTE 3D (PYDECK)
# ─────────────────────────────────────────────

def plot_trajectory_3d(df: pd.DataFrame, flight_id: str) -> pdk.Deck:
    """
    Génère une carte 3D de la trajectoire du vol en utilisant PyDeck.
    """
    g = _get_flight(df, flight_id)
    
    # Préparation des données pour PyDeck (exagération x5 pour la visu)
    g["alt_m_viz"] = (g["altitude_ft"] * 0.3048) * 5 
    
    path_data = pd.DataFrame({
        "path": [g[["lon", "lat", "alt_m_viz"]].values.tolist()]
    })
    
    # Paramétrage de la caméra initiale
    view_state = pdk.ViewState(
        latitude=g["lat"].mean(),
        longitude=g["lon"].mean(),
        zoom=9,
        pitch=45,
        bearing=0
    )
    
    # Création de la couche
    layer = pdk.Layer(
        "PathLayer",
        data=path_data,
        get_path="path",
        get_color=[249, 115, 22, 255], 
        width_scale=20,
        width_min_pixels=3,
        get_width=5,
    )
    
    # Assemblage de la carte
    deck = pdk.Deck(
        layers=[layer], 
        initial_view_state=view_state, 
        map_style="mapbox://styles/mapbox/satellite-v9",
        tooltip=False
    )
    
    return deck


# ─────────────────────────────────────────────
# 3. PROFIL D'ALTITUDE
# ─────────────────────────────────────────────

def plot_altitude_profile(
    df: pd.DataFrame,
    flight_id: str = None,
    save: bool = True,
) -> go.Figure:
    """
    Profil d'altitude en fonction de la distance cumulée.
    """
    _ensure_output_dir()

    if flight_id:
        flights = {flight_id: _get_flight(df, flight_id)}
    else:
        flights = {fid: _get_flight(df, fid) for fid in sorted(df["flight_id"].unique())}

    fig = go.Figure()

    for fid, g in flights.items():
        if "dist_cum_nm" not in g.columns:
            raise ValueError("compute_distance() doit être appelé avant plot_altitude_profile().")

        x = g["dist_cum_nm"].values
        y = g["altitude_ft"].values

        if "phase" in g.columns and len(flights) == 1:
            _add_phase_background(fig, g, x_col="dist_cum_nm", y_max=y.max() * 1.1)

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            name=f"Altitude — {fid}",
            line=dict(color=_flight_color(df, fid), width=2),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Distance : %{x:.1f} nm<br>"
                "Altitude : %{y} ft<br>"
                "Heure : %{customdata[1]}<br>"
                "Vitesse : %{customdata[2]} kt"
                "<extra></extra>"
            ),
            customdata=np.stack([
                g["flight_id"],
                g["datetime_utc"].dt.strftime("%H:%M UTC"),
                g["speed_kt"],
            ], axis=1),
        ))

    fig.update_layout(
        title="Profil d'altitude",
        xaxis_title="Distance cumulée (nm)",
        yaxis_title="Altitude (ft)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )

    if save:
        fname = OUTPUT_DIR / "altitude_profile.html"
        fig.write_html(str(fname))

    return fig

def _add_phase_background(fig: go.Figure, g: pd.DataFrame, x_col: str, y_max: float, row=None, col=None):
    """Ajoute des rectangles de fond colorés par phase sur un graphique Plotly."""
    x = g[x_col].values
    phases = g["phase"].values
    n = len(g)

    i = 0
    while i < n - 1:
        phase = phases[i]
        color = PHASE_COLORS.get(phase, "#38bdf8")
        j = i + 1
        while j < n and phases[j] == phase:
            j += 1

        kwargs = dict(
            x0=x[i], x1=x[min(j, n-1)],
            fillcolor=color,
            opacity=0.12,
            layer="below",
            line_width=0,
            annotation_text=PHASE_LABELS.get(phase, phase) if (j - i) > 30 else "",
            annotation_position="top left",
            annotation_font_size=10
        )
        if row is not None and col is not None:
            kwargs.update({"row": row, "col": col})
            
        fig.add_vrect(**kwargs)
        i = j


# ─────────────────────────────────────────────
# 4. PROFIL VITESSE + VARIO
# ─────────────────────────────────────────────

def plot_speed_vario(
    df: pd.DataFrame,
    flight_id: str,
    save: bool = True,
) -> go.Figure:
    _ensure_output_dir()

    if "vario_fpm" not in df.columns:
        raise ValueError("compute_vario() doit être appelé avant plot_speed_vario().")

    g = _get_flight(df, flight_id)
    t = g["datetime_utc"]

    fig = sp.make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Vitesse sol (kt)", "Vario lissé (ft/min)"),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.5],
    )

    fig.add_trace(go.Scatter(
        x=t, y=g["speed_kt"],
        mode="lines",
        name="Vitesse (kt)",
        line=dict(color="#38bdf8", width=2),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.15)",
    ), row=1, col=1)

    vario = g["vario_fpm"].values

    fig.add_trace(go.Scatter(
        x=t, y=np.where(vario > 0, vario, 0),
        mode="lines", name="Montée",
        line=dict(color="#f97316", width=0),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.35)",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=np.where(vario < 0, vario, 0),
        mode="lines", name="Descente",
        line=dict(color="#22c55e", width=0),
        fill="tozeroy", fillcolor="rgba(34,197,94,0.35)",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=vario,
        mode="lines", name="Vario",
        line=dict(color="#1e293b", width=1.5),
        showlegend=False,
    ), row=2, col=1)

    fig.add_shape(
        type="line", x0=0, x1=1, xref="paper", y0=0, y1=0, yref="y2",
        line=dict(color="black", width=1, dash="dot"),
        row=2, col=1
    )

    fig.update_layout(
        title=f"Vitesse & Vario — {flight_id}",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    
    if save:
        fname = OUTPUT_DIR / f"speed_vario_{flight_id}.html"
        fig.write_html(str(fname))

    return fig


# ─────────────────────────────────────────────
# 5. DASHBOARD D'UN VOL
# ─────────────────────────────────────────────

def plot_dashboard(
    df: pd.DataFrame,
    flight_id: str,
    save: bool = True,
) -> go.Figure:
    """Dashboard complet d'un vol sur une seule page."""
    _ensure_output_dir()
    from src.analysis import stats_per_flight

    g = _get_flight(df, flight_id)
    stats = stats_per_flight(df[df["flight_id"] == flight_id]).iloc[0]

    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Profil d'altitude",
            "Distribution des phases",
            "Vitesse sol (kt)",
            "Vario lissé (ft/min)",
        ),
        specs=[
            [{"type": "xy"}, {"type": "domain"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # ── Profil altitude ──
    if "dist_cum_nm" in g.columns:
        if "phase" in g.columns:
            _add_phase_background(fig, g, "dist_cum_nm", g["altitude_ft"].max() * 1.1, row=1, col=1)
                
        fig.add_trace(go.Scatter(
            x=g["dist_cum_nm"], y=g["altitude_ft"],
            mode="lines", name="Altitude",
            line=dict(color="#e63946", width=2), 
            showlegend=False,                    
        ), row=1, col=1)

    # ── Donut des phases ──
    if "phase" in g.columns:
        # Somme exacte des temps au lieu de la multiplication par la médiane
        dt_sec = g["datetime_utc"].diff().dt.total_seconds().fillna(0)
        phase_min = (dt_sec.groupby(g["phase"]).sum() / 60).round(1)

        fig.add_trace(go.Pie(
            labels=[PHASE_LABELS.get(p, p) for p in phase_min.index],
            values=phase_min.values,
            marker=dict(colors=[PHASE_COLORS.get(p, "#ccc") for p in phase_min.index]),
            hole=0.45,
            textinfo="label+percent",
            showlegend=False,
        ), row=1, col=2)

    # ── Vitesse ──
    fig.add_trace(go.Scatter(
        x=g["datetime_utc"], y=g["speed_kt"],
        mode="lines", name="Vitesse",
        line=dict(color="#38bdf8", width=1.5),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.15)",
        showlegend=False,
    ), row=2, col=1)

    # ── Vario ──
    if "vario_fpm" in g.columns:
        vario = g["vario_fpm"].values
        fig.add_trace(go.Scatter(
            x=g["datetime_utc"], y=np.where(vario > 0, vario, 0),
            fill="tozeroy", fillcolor="rgba(249,115,22,0.35)",
            line=dict(width=0), name="Montée", showlegend=False,
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=g["datetime_utc"], y=np.where(vario < 0, vario, 0),
            fill="tozeroy", fillcolor="rgba(34,197,94,0.35)",
            line=dict(width=0), name="Descente", showlegend=False,
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=g["datetime_utc"], y=vario,
            mode="lines", line=dict(color="#1e293b", width=1.2),
            name="Vario", showlegend=False,
        ), row=2, col=2)
        
        # Correction ligne de base Vario pour subplots mixtes
        fig.add_shape(
            type="line", x0=0, x1=1, xref="x3 domain", y0=0, y1=0, yref="y3",
            line=dict(color="black", width=1, dash="dot"),
            row=2, col=2
        )

    # ── Axes via Layout (Safe for Domain subplots) ──
    fig.update_layout(
        xaxis_title="Distance (nm)",
        yaxis_title="Altitude (ft)",
        xaxis2_title="Heure UTC",
        yaxis2_title="Vitesse (kt)",
        xaxis3_title="Heure UTC",
        yaxis3_title="Vario (ft/min)",
    )

    duration = f"{stats.get('duration_min', '?'):.0f} min" if pd.notna(stats.get('duration_min')) else "?"
    distance = f"{stats.get('dist_total_nm', '?'):.0f} nm" if pd.notna(stats.get('dist_total_nm')) else "?"
    alt_max  = f"{stats.get('alt_max_ft', '?')} ft" if pd.notna(stats.get('alt_max_ft')) else "?"
    wind     = f"{stats.get('wind_speed_avg_kt', '?'):.0f} kt" if pd.notna(stats.get('wind_speed_avg_kt')) else "?"

    title = (
        f"<b>Dashboard — {flight_id}</b>   |   "
        f"Durée : {duration}   •   Distance : {distance}   •   "
        f"Alt max : {alt_max}   •   Vent moy : {wind}"
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template="plotly_white",
        height=750,
    )

    if save:
        fname = OUTPUT_DIR / f"dashboard_{flight_id}.html"
        fig.write_html(str(fname))

    return fig


# ─────────────────────────────────────────────
# 6. COMPARAISON MULTI-VOLS
# ─────────────────────────────────────────────

def plot_comparison(
    df_summary: pd.DataFrame,
    save: bool = True,
) -> go.Figure:
    _ensure_output_dir()
    df_s = df_summary.sort_values("date")
    labels = df_s["flight_id"].values

    metrics = [
        ("duration_min",    "Durée (min)",             "#38bdf8"),
        ("dist_total_nm",   "Distance (nm)",            "#f97316"),
        ("alt_max_ft",      "Altitude max (ft)",        "#a855f7"),
        ("speed_cruise_kt", "Vitesse croisière (kt)",   "#22c55e"),
        ("roc_max_fpm",     "Taux de montée max (fpm)", "#f5c518"),
        ("fuel_liters",     "Carburant (L)",            "#e63946"),
    ]

    metrics = [m for m in metrics if m[0] in df_s.columns and df_s[m[0]].notna().any()]

    n = len(metrics)
    cols = 2
    rows = (n + 1) // cols

    fig = sp.make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[m[1] for m in metrics],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    for idx, (col, label, color) in enumerate(metrics):
        r = idx // cols + 1
        c = idx % cols + 1

        fig.add_trace(go.Bar(
            x=labels, y=df_s[col], name=label,
            marker_color=color, showlegend=False,
            text=df_s[col].round(1), textposition="outside",
        ), row=r, col=c)

    fig.update_layout(
        title="Comparaison multi-vols",
        template="plotly_white",
        height=300 * rows,
    )

    if save:
        fname = OUTPUT_DIR / "comparison.html"
        fig.write_html(str(fname))

    return fig


# ─────────────────────────────────────────────
# 7. ROSE DES VENTS
# ─────────────────────────────────────────────

def plot_wind_rose(
    df: pd.DataFrame,
    flight_id: str = None,
    save: bool = True,
) -> plt.Figure:
    _ensure_output_dir()

    if "wind_dir_deg" not in df.columns or "wind_speed_kt" not in df.columns:
        raise ValueError("compute_wind() doit être appelé avant plot_wind_rose().")

    data = df if flight_id is None else df[df["flight_id"] == flight_id]

    if "phase" in data.columns:
        data = data[data["phase"] == "cruise"]
    data = data[data["wind_speed_kt"].notna() & data["wind_dir_deg"].notna()]

    if data.empty:
        return None

    n_sectors = 12
    sector_size = 360 / n_sectors
    sectors = np.arange(0, 360, sector_size)

    wind_dir = data["wind_dir_deg"].values
    wind_spd = data["wind_speed_kt"].values

    sector_speeds = []
    for s in sectors:
        mask = (wind_dir >= s) & (wind_dir < s + sector_size)
        sector_speeds.append(wind_spd[mask].mean() if mask.any() else 0)

    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection="polar")

    theta = np.radians(sectors + sector_size / 2)
    ax.bar(
        theta, sector_speeds, width=np.radians(sector_size) * 0.85,
        bottom=0, color=plt.cm.YlOrRd(np.array(sector_speeds) / (max(sector_speeds) + 0.1)),
        alpha=0.85, edgecolor="white", linewidth=0.5,
    )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.set_ylabel("Vitesse (kt)", labelpad=15)

    title = f"Rose des vents estimés — "
    title += flight_id if flight_id else f"{df['flight_id'].nunique()} vol(s)"
    ax.set_title(title, pad=20, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        fname = OUTPUT_DIR / f"wind_rose{'_' + flight_id if flight_id else ''}.png"
        fig.savefig(str(fname), dpi=150, bbox_inches="tight")

    return fig


# ─────────────────────────────────────────────
# 8. PRÉCISION DU MODÈLE CARBURANT
# ─────────────────────────────────────────────

def plot_fuel_accuracy(
    df_summary: pd.DataFrame,
    save: bool = True,
) -> go.Figure:
    _ensure_output_dir()
    required = ["fuel_liters", "fuel_predicted"]
    for col in required:
        if col not in df_summary.columns:
            raise ValueError(f"Colonne '{col}' manquante.")

    # Ajout du .copy() pour éviter les avertissements Pandas
    df_s = df_summary.dropna(subset=required).copy()
    if df_s.empty:
        return None

    # --- LA CORRECTION EST ICI ---
    if "TDP" in df_s.columns:
        df_s["TDP"] = pd.to_numeric(df_s["TDP"], errors="coerce").fillna(0).astype(int)
    else:
        df_s["TDP"] = 0
    # -----------------------------

    real = df_s["fuel_liters"]
    predicted = df_s["fuel_predicted"]
    mae = (real - predicted).abs().mean()

    fig = go.Figure()

    # La ligne de prédiction parfaite (Diagonale)
    ref_min = min(real.min(), predicted.min()) * 0.9
    ref_max = max(real.max(), predicted.max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[ref_min, ref_max], y=[ref_min, ref_max],
        mode="lines", line=dict(color="gray", dash="dash", width=1),
        name="Prédiction parfaite", showlegend=True,
    ))

    # --- SÉPARATION NAV vs TDP POUR LES COULEURS ---
    mask_tdp = df_s["TDP"] == 1
    df_nav = df_s[~mask_tdp]
    df_tdp = df_s[mask_tdp]

    # Tracé 1 : Les Navigations (en Rouge)
    if not df_nav.empty:
        fig.add_trace(go.Scatter(
            x=df_nav["fuel_liters"], y=df_nav["fuel_predicted"], 
            mode="markers+text", text=df_nav["flight_id"], textposition="top center",
            marker=dict(size=10, color="#e63946", opacity=0.85),
            name="Navigations",
            hovertemplate="<b>%{text}</b><br>Réel : %{x:.1f} L<br>Prédit : %{y:.1f} L<br><extra></extra>"
        ))

    # Tracé 2 : Les Tours de Piste (en Bleu)
    if not df_tdp.empty:
        fig.add_trace(go.Scatter(
            x=df_tdp["fuel_liters"], y=df_tdp["fuel_predicted"], 
            mode="markers+text", text=df_tdp["flight_id"], textposition="top center",
            marker=dict(size=10, color="#2196f3", opacity=0.85), # Bleu
            name="Tours de Piste",
            hovertemplate="<b>%{text}</b><br>Réel : %{x:.1f} L<br>Prédit : %{y:.1f} L<br><extra></extra>"
        ))

    fig.update_layout(
        title=f"Précision du modèle carburant — MAE : {mae:.1f} L",
        xaxis_title="Carburant réel (L)",
        yaxis_title="Carburant prédit (L)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    if save:
        fname = OUTPUT_DIR / "fuel_accuracy.html"
        fig.write_html(str(fname))

    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.loader   import load_all_flights
    from src.analysis import enrich, stats_per_flight

    df = load_all_flights("data/")
    df = enrich(df)

    fid = df["flight_id"].iloc[0]
    df_summary = stats_per_flight(df)

    plot_trajectory(df, flight_id=fid)
    plot_trajectory_3d(df, flight_id=fid)
    plot_altitude_profile(df, flight_id=fid)
    plot_speed_vario(df, fid)
    plot_dashboard(df, fid)
    plot_comparison(df_summary)
    plot_wind_rose(df, fid)