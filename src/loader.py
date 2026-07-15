"""
loader.py — Flight Analyzer
============================
Charge et fusionne les données FlightRadar (CSV + KML) avec le carnet de vol
manuel (flights_log.csv) pour construire un DataFrame master propre.
 
Usage:
    from loader import load_all_flights, load_flight
 
    df = load_all_flights("data/")          # tous les vols
    df = load_flight("data/raw/2026-04-26_FGSBJ.csv",
                     "data/raw/2026-04-26_FGSBJ.kml")  # un seul vol
"""
 
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone
 
import pandas as pd
import numpy as np
 
 
# ─────────────────────────────────────────────
# 1. PARSING CSV
# ─────────────────────────────────────────────
 
def parse_csv(csv_path: str) -> pd.DataFrame:
    """
    Lit un CSV FlightRadar et retourne un DataFrame propre.
 
    Colonnes produites :
        timestamp_unix, datetime_utc, callsign,
        lat, lon, altitude_ft, speed_kt, heading_deg
    """
    df = pd.read_csv(csv_path)
 
    # Séparer "lat,lon" → deux colonnes numériques
    coords = df["Position"].str.split(",", expand=True)
    df["lat"] = coords[0].astype(float)
    df["lon"] = coords[1].astype(float)
 
    # Renommages propres
    df = df.rename(columns={
        "Timestamp":  "timestamp_unix",
        "UTC":        "datetime_utc",
        "Callsign":   "callsign",
        "Altitude":   "altitude_ft",
        "Speed":      "speed_kt",
        "Direction":  "heading_deg",
    })
 
    # Datetime aware (UTC)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
 
    # Supprimer la colonne Position brute
    df = df.drop(columns=["Position"])
 
    # Colonne d'index temporel pour les jointures
    df = df.sort_values("datetime_utc").reset_index(drop=True)
 
    return df[["timestamp_unix", "datetime_utc", "callsign",
               "lat", "lon", "altitude_ft", "speed_kt", "heading_deg"]]
 
 
# ─────────────────────────────────────────────
# 2. PARSING KML
# ─────────────────────────────────────────────
 
def _strip_ns(tag: str) -> str:
    """Retire le namespace XML d'un tag."""
    return tag.split("}")[-1] if "}" in tag else tag
 
 
def _parse_kml_metadata(doc_el) -> dict:
    """Extrait les métadonnées globales du document KML (avion, aéroclub...)."""
    meta = {}
    desc_el = None
 
    for child in doc_el:
        tag = _strip_ns(child.tag)
        if tag == "description":
            desc_el = child
        elif tag == "n":
            meta["kml_name"] = child.text
 
    if desc_el is not None and desc_el.text:
        html = desc_el.text
        # Appareil
        m = re.search(r"Robin[^<\"]+", html)
        if m:
            meta["aircraft"] = m.group(0).strip()
        # Aéroclub
        m = re.search(r"Aéroclub[^<]+", html, re.IGNORECASE)
        if m:
            meta["aeroclub"] = m.group(0).strip()
 
    return meta
 
 
def _parse_kml_trail(folder_el) -> pd.DataFrame:
    """
    Parse le Folder 'Trail' du KML (segments de trajectoire).
 
    Retourne un DataFrame avec, pour chaque segment :
        datetime_utc (borne début), altitude_m, altitude_ft_kml,
        speed_min_kt, speed_max_kt, color_argb
    """
    rows = []
    placemarks = [c for c in folder_el if _strip_ns(c.tag) == "Placemark"]
 
    for pm in placemarks:
        row = {}
        for child in pm:
            tag = _strip_ns(child.tag)
 
            if tag == "Style":
                for sub in child:
                    if _strip_ns(sub.tag) == "LineStyle":
                        for s2 in sub:
                            if _strip_ns(s2.tag) == "color":
                                row["color_argb"] = s2.text
 
            elif tag == "MultiGeometry":
                for sub in child:
                    if _strip_ns(sub.tag) == "LineString":
                        for s2 in sub:
                            if _strip_ns(s2.tag) == "coordinates":
                                # Format : "lon1,lat1,alt1 lon2,lat2,alt2"
                                pts = s2.text.strip().split()
                                if pts:
                                    parts = pts[0].split(",")
                                    if len(parts) == 3:
                                        row["alt_m_start"] = float(parts[2])
                                if len(pts) > 1:
                                    parts = pts[1].split(",")
                                    if len(parts) == 3:
                                        row["alt_m_end"] = float(parts[2])
 
            elif tag == "description":
                txt = re.sub(r"<[^>]+>", "", child.text or "")
                # Timestamp de début du segment
                ts_match = re.search(
                    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC)", txt
                )
                if ts_match:
                    row["datetime_utc"] = pd.to_datetime(
                        ts_match.group(1), utc=True
                    )
                # Vitesses (ex: "67 kt - 68 kt" ou "67 kt")
                spd = re.findall(r"(\d+)\s*kt", txt)
                if spd:
                    speeds = [int(s) for s in spd]
                    row["speed_min_kt"] = min(speeds)
                    row["speed_max_kt"] = max(speeds)
 
        if row:
            rows.append(row)
 
    df = pd.DataFrame(rows)
 
    if df.empty:
        return df
 
    # Altitude en pieds depuis les mètres (source KML = plus précise)
    if "alt_m_start" in df.columns:
        df["altitude_ft_kml"] = (df["alt_m_start"] * 3.28084).round(1)
 
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    return df
 
 
def parse_kml(kml_path: str) -> tuple[dict, pd.DataFrame]:
    """
    Lit un fichier KML FlightRadar.
 
    Retourne :
        metadata (dict)    — infos avion, aéroclub
        df_trail (DataFrame) — trajectoire 3D enrichie par segment
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()
    doc = root[0]
 
    metadata = _parse_kml_metadata(doc)
 
    # Trouver les deux folders
    folders = [c for c in doc if _strip_ns(c.tag) == "Folder"]
    df_trail = pd.DataFrame()
 
    if len(folders) >= 2:
        # Folder 2 = Trail (segments avec altitude 3D)
        df_trail = _parse_kml_trail(folders[1])
 
    return metadata, df_trail
 
 
# ─────────────────────────────────────────────
# 3. FUSION CSV + KML
# ─────────────────────────────────────────────
 
def _merge_kml_altitude(df_csv: pd.DataFrame, df_trail: pd.DataFrame) -> pd.DataFrame:
    """
    Jointure par timestamp entre le CSV et le Trail KML.
    Ajoute altitude_ft_kml et color_argb au DataFrame CSV.
    Méthode : merge_asof (tolérance ±5 secondes).
    """
    if df_trail.empty or "datetime_utc" not in df_trail.columns:
        return df_csv
 
    trail_slim = df_trail[
        [c for c in ["datetime_utc", "altitude_ft_kml", "color_argb"]
         if c in df_trail.columns]
    ].copy()
 
    df_merged = pd.merge_asof(
        df_csv.sort_values("datetime_utc"),
        trail_slim.sort_values("datetime_utc"),
        on="datetime_utc",
        tolerance=pd.Timedelta("5s"),
        direction="nearest",
    )
 
    return df_merged
 
 
# ─────────────────────────────────────────────
# 4. CARNET DE VOL (flights_log.csv)
# ─────────────────────────────────────────────

def _load_flights_log(data_dir: str) -> pd.DataFrame:
    """
    Charge data/flights_log.csv s'il existe.
    Colonnes attendues : date, flight_id, fuel_liters, engine_time_h, departure, arrival, is_mine, TDP, notes
    """
    log_path = Path(data_dir) / "flights_log.csv"
    if not log_path.exists():
        return pd.DataFrame()

    # skipinitialspace=True est crucial pour éviter le bug des espaces dans les CSV
    df = pd.read_csv(log_path, skipinitialspace=True)
    
    # Nettoyage de sécurité
    df.columns = df.columns.str.strip()
    if "flight_id" in df.columns:
        df["flight_id"] = df["flight_id"].astype(str).str.strip()
        
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    # --- LA CORRECTION EST ICI ---
    if "TDP" in df.columns:
        df["TDP"] = pd.to_numeric(df["TDP"], errors="coerce").fillna(0).astype(int)
    # -----------------------------
        
    return df
 
 
# ─────────────────────────────────────────────
# 5. CHARGEMENT D'UN VOL
# ─────────────────────────────────────────────

def load_flight(csv_path: str, kml_path: str = None) -> tuple[pd.DataFrame, dict]:
    """
    Charge et fusionne un vol complet.

    Paramètres :
        csv_path   — chemin vers le CSV FlightRadar
        kml_path   — chemin vers le KML (optionnel mais recommandé)

    Retourne :
        df       — DataFrame du vol enrichi
        metadata — dict avec infos avion, aéroclub, flight_id
    """
    df = parse_csv(csv_path)

    # L'identifiant du vol EST désormais exactement le nom du fichier sans l'extension
    # ex: "2026-05-07_1430_FGSRR"
    stem = Path(csv_path).stem  
    flight_id = stem

    # On extrait la date (les 10 premiers caractères : YYYY-MM-DD)
    flight_date = stem[:10] if len(stem) >= 10 else "unknown"

    df["flight_id"] = flight_id
    
    try:
        df["flight_date"] = pd.to_datetime(flight_date).date()
    except:
        df["flight_date"] = pd.NaT

    metadata = {"flight_id": flight_id, "flight_date": flight_date}

    # Enrichissement KML
    if kml_path and Path(kml_path).exists():
        kml_meta, df_trail = parse_kml(kml_path)
        metadata.update(kml_meta)
        df = _merge_kml_altitude(df, df_trail)

    return df, metadata
 
 
# ─────────────────────────────────────────────
# 6. CHARGEMENT DE TOUS LES VOLS
# ─────────────────────────────────────────────

def load_all_flights(data_dir: str = "data/") -> pd.DataFrame:
    """
    Scanne data/raw/, détecte les paires CSV+KML et charge tous les vols.
    Si le dossier est vide, retourne un DataFrame vide au lieu de planter l'application.
    """
    raw_dir = Path(data_dir) / "raw"
    
    # 1. Si le dossier n'existe pas, on le crée silencieusement et on retourne vide
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        return pd.DataFrame()

    # Détecter tous les CSV dans raw/
    csv_files = sorted(raw_dir.glob("*.csv"))
    
    # 2. S'il n'y a pas de CSV, on retourne vide (plus d'erreur fatale ici !)
    if not csv_files:
        return pd.DataFrame()

    all_dfs = []
    all_metadata = []

    for csv_path in csv_files:
        kml_path = csv_path.with_suffix(".kml")
        if not kml_path.exists():
            stem = csv_path.stem
            candidates = list(raw_dir.glob(f"{stem}*.kml"))
            if candidates:
                kml_path = candidates[0]
            else:
                kml_path = None

        try:
            df, meta = load_flight(str(csv_path), str(kml_path) if kml_path else None)
            all_dfs.append(df)
            all_metadata.append(meta)
        except Exception as e:
            print(f" → ERREUR sur {csv_path.name} : {e}")

    # 3. Si aucun vol n'a réussi à charger, on retourne vide (plus d'erreur fatale !)
    if not all_dfs:
        return pd.DataFrame()

    # Concaténation master
    df_master = pd.concat(all_dfs, ignore_index=True)

    # Jointure avec le carnet de vol
    df_log = _load_flights_log(data_dir)
    if not df_log.empty:
        df_master = df_master.merge(
            df_log.drop(columns=["date"], errors="ignore"),
            on="flight_id",
            how="left",
        )

    return df_master
 
 
# ─────────────────────────────────────────────
# 7. UTILITAIRES
# ─────────────────────────────────────────────
 
def flight_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un résumé par vol (une ligne par flight_id).
    Utile pour avoir une vue rapide de ta base de données.
    """
    def summarize(g):
        duration = (g["datetime_utc"].max() - g["datetime_utc"].min())
        return pd.Series({
            "date":           g["flight_date"].iloc[0],
            "callsign":       g["callsign"].iloc[0],
            "n_points":       len(g),
            "duration_min":   round(duration.total_seconds() / 60, 1),
            "alt_max_ft":     g["altitude_ft"].max(),
            "speed_max_kt":   g["speed_kt"].max(),
            "departure":      g.get("departure", pd.Series([None])).iloc[0],
            "arrival":        g.get("arrival", pd.Series([None])).iloc[0],
            "fuel_liters":    g.get("fuel_liters", pd.Series([None])).iloc[0],
        })
 
    return df.groupby("flight_id").apply(summarize).reset_index()
 
 
# ─────────────────────────────────────────────
# POINT D'ENTRÉE (test rapide)
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    import sys
 
    if len(sys.argv) == 3:
        # Test sur un seul vol : python loader.py chemin.csv chemin.kml
        csv_p, kml_p = sys.argv[1], sys.argv[2]
        df, meta = load_flight(csv_p, kml_p)
        print("\nMétadonnées :", meta)
        print("\nAperçu du DataFrame :")
        print(df.head(5).to_string())
        print(f"\nColonnes : {df.columns.tolist()}")
        print(f"Shape : {df.shape}")
    else:
        # Test sur tous les vols
        df_all = load_all_flights("data/")
        print("\nRésumé des vols :")
        print(flight_summary(df_all).to_string())