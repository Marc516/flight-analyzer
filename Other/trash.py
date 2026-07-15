"""
plots.py — Flight Analyzer
===========================
Toutes les visualisations du projet.

Fonctions publiques :
    plot_trajectory(df, flight_id=None)          → carte Folium (HTML)
    plot_altitude_profile(df, flight_id=None)    → profil altitude Plotly
    plot_speed_vario(df, flight_id)              → vitesse + vario Plotly
    plot_dashboard(df, flight_id)                → dashboard complet un vol
    plot_comparison(df_summary)                  → comparaison multi-vols
    plot_wind_rose(df, flight_id=None)           → rose des vents
    plot_fuel_accuracy(df_summary)               → précision modèle carburant

Couleurs des phases (convention projet) :
    ground   → gris
    takeoff  → jaune
    climb    → orange
    cruise   → bleu clair
    descent  → vert
    landing  → violet
"""

import os
import numpy as np
import pandas as pd
import folium
import pydeck as pdk
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
from pathlib import Path


# ─────────────────────────────────────────────
# CONSTANTES VISUELLES
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

# Palette pour différencier les vols (multi-vols)
FLIGHT_PALETTE = [
    "#e63946", "#2196f3", "#ff9800", "#4caf50",
    "#9c27b0", "#00bcd4", "#ff5722", "#607d8b",
]

# Dossier de sortie des fichiers HTML/PNG
OUTPUT_DIR = Path("outputs")


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
# TUILES IGN — CARTE OACI 1/500 000
# ─────────────────────────────────────────────

# Carte aéronautique officielle IGN Géoportail (SCAN OACI)
# Accès gratuit via la clé publique "essentiels"
# Doc : https://geoservices.ign.fr/documentation/services/api-et-services-ogc/images-tuilees-wmts-ogc
IGN_OACI_TILES = (
    "https://wxs.ign.fr/essentiels/geoportail/wmts"
    "?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
    "&LAYER=GEOGRAPHICALGRIDSYSTEMS.MAPS.SCAN-OACI"
    "&STYLE=normal&FORMAT=image/png"
    "&TILEMATRIXSET=PM"
    "&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}"
)

IGN_OACI_ATTR = (
    "© <a href='https://www.geoportail.gouv.fr/'>IGN Géoportail</a> — "
    "Carte aéronautique OACI 1/500 000"
)


# ─────────────────────────────────────────────
# 1. CARTE INTERACTIVE (FOLIUM)
# ─────────────────────────────────────────────

def plot_trajectory(
    df: pd.DataFrame,
    flight_id: str = None,
    color_by: str = "phase",   # "phase" | "altitude" | "speed" | "flight"
    save: bool = True,
) -> folium.Map:
    """
    Carte interactive de la trajectoire sur fond carte OACI IGN 1/500 000.

    Paramètres :
        df         — DataFrame master enrichi
        flight_id  — None = tous les vols, sinon un seul
        color_by   — variable de coloration du tracé
        save       — sauvegarde le HTML dans outputs/

    Retourne : objet folium.Map (affichable dans Jupyter avec display(m))
    """
    _ensure_output_dir()

    # Filtrage
    if flight_id:
        flights = {flight_id: _get_flight(df, flight_id)}
    else:
        flights = {fid: _get_flight(df, fid) for fid in sorted(df["flight_id"].unique())}

    # Centre de la carte sur le premier vol
    first = next(iter(flights.values()))
    center = _center(first)

    # Création de la carte avec fond OACI
    m = folium.Map(
        location=center,
        zoom_start=9,
        tiles=IGN_OACI_TILES,
        attr=IGN_OACI_ATTR,
    )

    # Couche OSM en alternatif (bouton de sélection)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles=IGN_OACI_TILES,
        attr=IGN_OACI_ATTR,
        name="Carte OACI IGN",
    ).add_to(m)

    for fid, g in flights.items():
        coords = list(zip(g["lat"], g["lon"]))

        if color_by == "phase" and "phase" in g.columns:
            # Tracé segment par segment coloré par phase
            _add_phase_polyline(m, g, fid)

        elif color_by == "altitude":
            _add_colored_polyline(m, g, "altitude_ft", fid)

        elif color_by == "speed":
            _add_colored_polyline(m, g, "speed_kt", fid)

        else:
            # Couleur unique par vol
            color = _flight_color(df, fid)
            folium.PolyLine(
                coords,
                color=color,
                weight=3,
                opacity=0.85,
                tooltip=fid,
            ).add_to(m)

        # Marqueur de départ
        folium.Marker(
            location=coords[0],
            icon=folium.Icon(color="green", icon="plane", prefix="fa"),
            popup=f"<b>Départ</b><br>{fid}<br>{g['datetime_utc'].iloc[0].strftime('%H:%M UTC')}",
        ).add_to(m)

        # Marqueur d'arrivée
        folium.Marker(
            location=coords[-1],
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
            popup=f"<b>Arrivée</b><br>{fid}<br>{g['datetime_utc'].iloc[-1].strftime('%H:%M UTC')}",
        ).add_to(m)

        # Popups sur les points (tous les 50 points pour ne pas surcharger)
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

    # Légende des phases
    if color_by == "phase":
        _add_phase_legend(m)

    folium.LayerControl().add_to(m)

    if save:
        fname = OUTPUT_DIR / f"trajectory_{'_'.join(flights.keys())}.html"
        m.save(str(fname))
        print(f"  Carte sauvegardée : {fname}")

    return m


def plot_trajectory_3d(df: pd.DataFrame, flight_id: str) -> pdk.Deck:
    """
    Génère une carte 3D de la trajectoire du vol en utilisant PyDeck.
    """
    g = _get_flight(df, flight_id)
    
    # 1. Préparation des données pour PyDeck
    # PyDeck prend l'altitude en mètres. 
    # ASTUCE VISUELLE : On multiplie l'altitude par 5 pour "exagérer" le relief en 3D 
    # et mieux voir les phases de montée/descente à cette échelle géographique.
    g["alt_m_viz"] = (g["altitude_ft"] * 0.3048) * 5 
    
    # PyDeck attend une seule ligne avec la liste des coordonnées [lon, lat, alt]
    path_data = pd.DataFrame({
        "path": [g[["lon", "lat", "alt_m_viz"]].values.tolist()]
    })
    
    # 2. Paramétrage de la caméra initiale (vue inclinée)
    view_state = pdk.ViewState(
        latitude=g["lat"].mean(),
        longitude=g["lon"].mean(),
        zoom=9,
        pitch=45,    # Inclinaison de la caméra (pour voir la 3D)
        bearing=0    # Cap de la caméra (0 = Nord)
    )
    
    # 3. Création de la couche "Chemin" (trajectoire)
    layer = pdk.Layer(
        "PathLayer",
        data=path_data,
        get_path="path",
        get_color=[249, 115, 22, 255], # Couleur orange aéro (RGBA)
        width_scale=20,
        width_min_pixels=3,
        get_width=5,
    )
    
    # 4. Assemblage de la carte (avec un fond satellite sombre très classe en 3D)
    deck = pdk.Deck(
        layers=[layer], 
        initial_view_state=view_state, 
        map_style="mapbox://styles/mapbox/satellite-v9",
        tooltip=False
    )
    
    return deck


def _add_phase_polyline(m: folium.Map, g: pd.DataFrame, fid: str):
    """Trace la trajectoire segment par segment, coloré par phase."""
    phases = g["phase"].values
    lats   = g["lat"].values
    lons   = g["lon"].values

    i = 0
    while i < len(g) - 1:
        phase = phases[i]
        color = PHASE_COLORS.get(phase, "#38bdf8")

        # Grouper les segments consécutifs de même phase
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
# 2. PROFIL D'ALTITUDE
# ─────────────────────────────────────────────

def plot_altitude_profile(
    df: pd.DataFrame,
    flight_id: str = None,
    save: bool = True,
) -> go.Figure:
    """
    Profil d'altitude en fonction de la distance cumulée.
    Fond coloré par phase (orange=montée, vert=descente, bleu=croisière).
    Plusieurs vols superposables.
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

        # Fond coloré par phase
        if "phase" in g.columns and len(flights) == 1:
            _add_phase_background(fig, g, x_col="dist_cum_nm", y_max=y.max() * 1.1)

        # Tracé de l'altitude
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
        print(f"  Profil altitude sauvegardé : {fname}")

    return fig


def _add_phase_background(fig: go.Figure, g: pd.DataFrame, x_col: str, y_max: float, row: int = "all", col: int = "all"):
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

        fig.add_vrect(
            x0=x[i], x1=x[min(j, n-1)],
            fillcolor=color,
            opacity=0.12,
            layer="below",
            line_width=0,
            annotation_text=PHASE_LABELS.get(phase, phase) if (j - i) > 30 else "",
            annotation_position="top left",
            annotation_font_size=10,
            row=row,
            col=col
        )
        i = j


# ─────────────────────────────────────────────
# 3. PROFIL VITESSE + VARIO
# ─────────────────────────────────────────────

def plot_speed_vario(
    df: pd.DataFrame,
    flight_id: str,
    save: bool = True,
) -> go.Figure:
    """
    Deux graphiques verticaux partageant l'axe X (temps) :
        - Haut : vitesse sol (kt)
        - Bas  : vario lissé (ft/min), zones + / - colorées
    """
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

    # ── Vitesse
    fig.add_trace(go.Scatter(
        x=t, y=g["speed_kt"],
        mode="lines",
        name="Vitesse (kt)",
        line=dict(color="#38bdf8", width=2),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.15)",
    ), row=1, col=1)

    # ── Vario : zones positive (orange) et négative (verte)
    vario = g["vario_fpm"].values

    # Zone positive — montée (orange)
    fig.add_trace(go.Scatter(
        x=t, y=np.where(vario > 0, vario, 0),
        mode="lines",
        name="Montée",
        line=dict(color="#f97316", width=0),
        fill="tozeroy",
        fillcolor="rgba(249,115,22,0.35)",
        showlegend=True,
    ), row=2, col=1)

    # Zone négative — descente (verte)
    fig.add_trace(go.Scatter(
        x=t, y=np.where(vario < 0, vario, 0),
        mode="lines",
        name="Descente",
        line=dict(color="#22c55e", width=0),
        fill="tozeroy",
        fillcolor="rgba(34,197,94,0.35)",
        showlegend=True,
    ), row=2, col=1)

    # Ligne du vario complet par-dessus
    fig.add_trace(go.Scatter(
        x=t, y=vario,
        mode="lines",
        name="Vario",
        line=dict(color="#1e293b", width=1.5),
        showlegend=False,
    ), row=2, col=1)

    # Ligne zéro
    fig.add_hline(y=0, line=dict(color="black", width=1, dash="dot"), row=2, col=1)

    fig.update_layout(
        title=f"Vitesse & Vario — {flight_id}",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Vitesse (kt)", row=1, col=1)
    fig.update_yaxes(title_text="Vario (ft/min)", row=2, col=1)
    fig.update_xaxes(title_text="Heure UTC", row=2, col=1)

    if save:
        fname = OUTPUT_DIR / f"speed_vario_{flight_id}.html"
        fig.write_html(str(fname))
        print(f"  Vitesse/Vario sauvegardé : {fname}")

    return fig


# ─────────────────────────────────────────────
# 4. DASHBOARD D'UN VOL
# ─────────────────────────────────────────────

def plot_dashboard(
    df: pd.DataFrame,
    flight_id: str,
    save: bool = True,
) -> go.Figure:
    """
    Dashboard complet d'un vol sur une seule page.
    """
    _ensure_output_dir()

    from src.analysis import stats_per_flight

    g = _get_flight(df, flight_id)
    stats = stats_per_flight(df[df["flight_id"] == flight_id]).iloc[0]

    # Déclaration des subplots
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

    # ── Profil altitude (haut gauche - row=1, col=1) ──
    if "dist_cum_nm" in g.columns:
        # Ajout des fonds colorés par phase directement ici !
        if "phase" in g.columns:
            x_vals = g["dist_cum_nm"].values
            phases = g["phase"].values
            n = len(g)
            i = 0
            while i < n - 1:
                phase = phases[i]
                color = PHASE_COLORS.get(phase, "#38bdf8")
                j = i + 1
                while j < n and phases[j] == phase:
                    j += 1
                fig.add_vrect(
                    x0=x_vals[i], x1=x_vals[min(j, n-1)],
                    fillcolor=color, opacity=0.12, layer="below", line_width=0,
                    row=1, col=1 # Strictement confiné à 1,1
                )
                i = j
                
        # Ligne de l'altitude
        fig.add_trace(go.Scatter(
            x=g["dist_cum_nm"], y=g["altitude_ft"],
            mode="lines", name="Altitude",
            line=dict(color=_flight_color(df, flight_id), width=2),
        ), row=1, col=1)

    # ── Donut des phases (haut droite - row=1, col=2) ──
    if "phase" in g.columns:
        phase_counts = g["phase"].value_counts()
        dt_med = g["datetime_utc"].diff().dt.total_seconds().median()
        phase_min = (phase_counts * dt_med / 60).round(1)

        fig.add_trace(go.Pie(
            labels=[PHASE_LABELS.get(p, p) for p in phase_min.index],
            values=phase_min.values,
            marker=dict(colors=[PHASE_COLORS.get(p, "#ccc") for p in phase_min.index]),
            hole=0.45,
            textinfo="label+percent",
            showlegend=False,
        ), row=1, col=2) 

    # ── Vitesse (bas gauche - row=2, col=1) ──
    fig.add_trace(go.Scatter(
        x=g["datetime_utc"], y=g["speed_kt"],
        mode="lines", name="Vitesse",
        line=dict(color="#38bdf8", width=1.5),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.15)",
        showlegend=False,
    ), row=2, col=1)

    # ── Vario (bas droite - row=2, col=2) ──
    if "vario_fpm" in g.columns:
        vario = g["vario_fpm"].values
        fig.add_trace(go.Scatter(
            x=g["datetime_utc"],
            y=np.where(vario > 0, vario, 0),
            fill="tozeroy", fillcolor="rgba(249,115,22,0.35)",
            line=dict(width=0), name="Montée", showlegend=False,
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=g["datetime_utc"],
            y=np.where(vario < 0, vario, 0),
            fill="tozeroy", fillcolor="rgba(34,197,94,0.35)",
            line=dict(width=0), name="Descente", showlegend=False,
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=g["datetime_utc"], y=vario,
            mode="lines", line=dict(color="#1e293b", width=1.2),
            name="Vario", showlegend=False,
        ), row=2, col=2)
        fig.add_shape(
            type="line",
            x0=0, x1=1, xref="x domain",  # S'étend sur toute la largeur du graphique ciblé
            y0=0, y1=0, yref="y",
            line=dict(color="black", width=1, dash="dot"),
            row=2, col=2
)

    # ── Mise à jour stricte des axes via layout (contourne le bug Plotly) ──
    fig.update_layout(
        xaxis_title="Distance (nm)",
        yaxis_title="Altitude (ft)",
        
        xaxis2_title="Heure UTC",
        yaxis2_title="Vitesse (kt)",
        
        xaxis3_title="Heure UTC",
        yaxis3_title="Vario (ft/min)",
    )

    # ── Titre et Layout globaux ──
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
        print(f"  Dashboard sauvegardé : {fname}")

    return fig


# ─────────────────────────────────────────────
# 5. COMPARAISON MULTI-VOLS
# ─────────────────────────────────────────────

def plot_comparison(
    df_summary: pd.DataFrame,
    save: bool = True,
) -> go.Figure:
    """
    Graphiques de comparaison multi-vols (un bar chart par métrique clé).
    Idéal pour voir la progression dans le temps.

    Métriques : durée, distance, altitude max, vitesse croisière,
                taux de montée max, carburant (si disponible).
    """
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

    # Ne garder que les métriques disponibles et non-vides
    metrics = [
        (col, label, color) for col, label, color in metrics
        if col in df_s.columns and df_s[col].notna().any()
    ]

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
            x=labels,
            y=df_s[col],
            name=label,
            marker_color=color,
            showlegend=False,
            text=df_s[col].round(1),
            textposition="outside",
        ), row=r, col=c)

    fig.update_layout(
        title="Comparaison multi-vols",
        template="plotly_white",
        height=300 * rows,
    )

    if save:
        fname = OUTPUT_DIR / "comparison.html"
        fig.write_html(str(fname))
        print(f"  Comparaison sauvegardée : {fname}")

    return fig


# ─────────────────────────────────────────────
# 6. ROSE DES VENTS
# ─────────────────────────────────────────────

def plot_wind_rose(
    df: pd.DataFrame,
    flight_id: str = None,
    save: bool = True,
) -> plt.Figure:
    """
    Rose des vents estimés en croisière.
    Chaque barre représente un secteur de 30° avec la force moyenne.
    Matplotlib projection polaire.
    """
    _ensure_output_dir()

    if "wind_dir_deg" not in df.columns or "wind_speed_kt" not in df.columns:
        raise ValueError("compute_wind() doit être appelé avant plot_wind_rose().")

    # Filtrage
    data = df if flight_id is None else df[df["flight_id"] == flight_id]

    # Garder uniquement les points en croisière avec vent valide
    if "phase" in data.columns:
        data = data[data["phase"] == "cruise"]
    data = data[data["wind_speed_kt"].notna() & data["wind_dir_deg"].notna()]

    if data.empty:
        print("  Aucune donnée de vent disponible.")
        return None

    # Secteurs de 30°
    n_sectors = 12
    sector_size = 360 / n_sectors
    sectors = np.arange(0, 360, sector_size)

    # Pour chaque secteur : vitesse moyenne
    wind_dir = data["wind_dir_deg"].values
    wind_spd = data["wind_speed_kt"].values

    sector_speeds = []
    for s in sectors:
        mask = (wind_dir >= s) & (wind_dir < s + sector_size)
        sector_speeds.append(wind_spd[mask].mean() if mask.any() else 0)

    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection="polar")

    # Plotly polaire : 0° = Nord, sens horaire
    theta = np.radians(sectors + sector_size / 2)
    bars  = ax.bar(
        theta,
        sector_speeds,
        width=np.radians(sector_size) * 0.85,
        bottom=0,
        color=plt.cm.YlOrRd(np.array(sector_speeds) / (max(sector_speeds) + 0.1)),
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    # Convention météo : 0° = Nord, sens horaire
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
        print(f"  Rose des vents sauvegardée : {fname}")

    return fig


# ─────────────────────────────────────────────
# 7. PRÉCISION DU MODÈLE CARBURANT
# ─────────────────────────────────────────────

def plot_fuel_accuracy(
    df_summary: pd.DataFrame,
    save: bool = True,
) -> go.Figure:
    """
    Scatter : carburant réel vs carburant prédit par le modèle.
    La diagonale = prédiction parfaite.
    Affiche aussi l'erreur moyenne absolue (MAE).

    Nécessite que fuel.py ait ajouté une colonne 'fuel_predicted'
    à df_summary.
    """
    _ensure_output_dir()

    required = ["fuel_liters", "fuel_predicted"]
    for col in required:
        if col not in df_summary.columns:
            raise ValueError(
                f"Colonne '{col}' manquante. "
                "Lancez fuel.predict_and_evaluate() avant ce graphique."
            )

    df_s = df_summary.dropna(subset=required)
    if df_s.empty:
        print("  Pas assez de données carburant pour ce graphique.")
        return None

    real      = df_s["fuel_liters"]
    predicted = df_s["fuel_predicted"]
    mae = (real - predicted).abs().mean()

    fig = go.Figure()

    # Diagonale de référence
    ref_min = min(real.min(), predicted.min()) * 0.9
    ref_max = max(real.max(), predicted.max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[ref_min, ref_max], y=[ref_min, ref_max],
        mode="lines",
        line=dict(color="gray", dash="dash", width=1),
        name="Prédiction parfaite",
        showlegend=True,
    ))

    # Points
    fig.add_trace(go.Scatter(
        x=real, y=predicted,
        mode="markers+text",
        text=df_s["flight_id"],
        textposition="top center",
        marker=dict(size=10, color="#e63946", opacity=0.85),
        name="Vols",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Réel : %{x:.1f} L<br>"
            "Prédit : %{y:.1f} L<br>"
            "<extra></extra>"
        ),
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
        print(f"  Précision carburant sauvegardée : {fname}")

    return fig


# ─────────────────────────────────────────────
# POINT D'ENTRÉE (test rapide)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.loader   import load_all_flights
    from src.analysis import enrich, stats_per_flight

    print("Chargement et enrichissement...")
    df = load_all_flights("data/")
    df = enrich(df)

    fid = df["flight_id"].iloc[0]
    df_summary = stats_per_flight(df)

    print(f"\nGénération des visualisations pour : {fid}")

    plot_trajectory(df, flight_id=fid)
    plot_altitude_profile(df, flight_id=fid)
    plot_speed_vario(df, fid)
    plot_dashboard(df, fid)
    plot_comparison(df_summary)
    plot_wind_rose(df, fid)

    print(f"\n✓ Tous les fichiers générés dans le dossier outputs/")