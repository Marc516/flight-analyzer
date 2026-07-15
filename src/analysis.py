"""
analysis.py — Flight Analyzer
==============================
Calculs dérivés sur le DataFrame master produit par loader.py.
 
Fonctions publiques :
    enrich(df)                → pipeline complet (appelle tout dans l'ordre)
    compute_vario(df)         → taux de montée/descente lissé (ft/min)
    compute_distance(df)      → distance GPS par segment + cumulée (nm)
    detect_phases(df)         → phase de vol pour chaque point
    compute_wind(df)          → estimation du vent (direction + force)
    stats_per_flight(df)      → résumé statistique par flight_id
 
Usage :
    from src.loader   import load_all_flights
    from src.analysis import enrich, stats_per_flight
 
    df = load_all_flights("data/")
    df = enrich(df)
    print(stats_per_flight(df))
"""
 
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2, degrees
from sklearn.linear_model import LinearRegression
 
 
# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
 
# Seuils calibrés sur DR-400 / données FlightRadar
TAXI_SPEED_MAX_KT      = 30     # en dessous = roulage ou arrêt
AIRBORNE_ALT_MIN_FT    = 50     # en dessous = considéré au sol
CLIMB_VARIO_MIN_FPM    = 150    # vario lissé > seuil → montée
DESCENT_VARIO_MAX_FPM  = -150   # vario lissé < seuil → descente
VARIO_SMOOTH_WINDOW    = 15     # fenêtre de lissage du vario (en points, ~30s)
PHASE_SMOOTH_WINDOW    = 10     # fenêtre de lissage des phases (évite les micro-transitions)
 
# Phases (labels)
PHASE_GROUND   = "ground"
PHASE_TAKEOFF  = "takeoff"
PHASE_CLIMB    = "climb"
PHASE_CRUISE   = "cruise"
PHASE_DESCENT  = "descent"
PHASE_LANDING  = "landing"
 
 
# ─────────────────────────────────────────────
# 1. VARIO (taux de montée/descente)
# ─────────────────────────────────────────────
 
def compute_vario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le taux de montée/descente en ft/min pour chaque point.
 
    Deux colonnes ajoutées :
        vario_raw_fpm    — vario instantané brut (très bruité, altitude arrondie à 100ft)
        vario_fpm        — vario lissé (moyenne glissante, utilisé pour la détection de phases)
 
    Le lissage est essentiel : l'altitude FlightRadar est arrondie à 100 ft,
    ce qui génère des vario instantanés aberrants (±3000 ft/min) alors que
    le DR-400 monte max ~700 ft/min. On utilise une fenêtre de 15 points (~30s).
 
    Gestion des trous de données : on remet le vario à NaN quand l'intervalle
    de temps entre deux points dépasse 30 secondes (couverture FlightRadar perdue).
    """
    df = df.copy()
 
    results = []
 
    for flight_id, group in df.groupby("flight_id", sort=False):
        g = group.sort_values("datetime_utc").copy()
 
        dt_sec = g["datetime_utc"].diff().dt.total_seconds()
        dalt   = g["altitude_ft"].diff()
 
        # Vario brut en ft/min
        vario_raw = (dalt / dt_sec * 60)
 
        # Invalider les vario calculés sur un trou de données
        vario_raw[dt_sec > 30] = np.nan
 
        # Lissage par moyenne glissante centrée
        vario_smooth = (
            vario_raw
            .rolling(window=VARIO_SMOOTH_WINDOW, center=True, min_periods=3)
            .mean()
            .round(1)
        )
 
        g["vario_raw_fpm"] = vario_raw.round(1)
        g["vario_fpm"]     = vario_smooth
 
        results.append(g)
 
    return pd.concat(results).sort_index()
 
 
# ─────────────────────────────────────────────
# 2. DISTANCE GPS
# ─────────────────────────────────────────────
 
def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Distance en nautical miles entre deux points GPS.
    Formule de Haversine — précision suffisante pour des distances < 500 nm.
    """
    R_NM = 3440.065  # rayon terrestre en nm
 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return R_NM * 2 * atan2(sqrt(a), sqrt(1 - a))
 
 
def compute_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la distance GPS entre points consécutifs et la distance cumulée.
 
    Deux colonnes ajoutées :
        dist_nm      — distance du segment précédent (nm)
        dist_cum_nm  — distance cumulée depuis le début du vol (nm)
 
    Les segments sur un trou de données (> 30s) sont gardés mais marqués
    NaN pour dist_nm afin de ne pas fausser la distance cumulée.
    """
    df = df.copy()
 
    results = []
 
    for flight_id, group in df.groupby("flight_id", sort=False):
        g = group.sort_values("datetime_utc").copy()
 
        dt_sec = g["datetime_utc"].diff().dt.total_seconds()
 
        lats = g["lat"].values
        lons = g["lon"].values
        n = len(g)
 
        dist = np.zeros(n)
        dist[0] = 0.0
 
        for i in range(1, n):
            if dt_sec.iloc[i] > 30:
                dist[i] = np.nan  # trou de données
            else:
                dist[i] = _haversine_nm(lats[i-1], lons[i-1], lats[i], lons[i])
 
        g["dist_nm"]     = dist
        g["dist_cum_nm"] = np.nancumsum(dist).round(3)
 
        results.append(g)
 
    return pd.concat(results).sort_index()
 
 
# ─────────────────────────────────────────────
# 3. DÉTECTION DES PHASES
# ─────────────────────────────────────────────
 
def _assign_phase(row) -> str:
    """
    Règles de détection de phase pour un point donné.
    Appelée ligne par ligne après calcul du vario lissé.
 
    Ordre des règles : du plus contraignant au plus général.
    """
    alt   = row["altitude_ft"]
    speed = row["speed_kt"]
    vario = row["vario_fpm"]  # peut être NaN
 
    # --- Au sol ---
    if alt < AIRBORNE_ALT_MIN_FT:
        if speed < TAXI_SPEED_MAX_KT:
            return PHASE_GROUND
        else:
            return PHASE_TAKEOFF  # vitesse sol élevée + altitude nulle = roulage décollage
 
    # --- En vol ---
    if pd.isna(vario):
        return PHASE_CRUISE  # pas de vario dispo → on suppose croisière
 
    if vario > CLIMB_VARIO_MIN_FPM:
        return PHASE_CLIMB
    elif vario < DESCENT_VARIO_MAX_FPM:
        if alt < 500:
            return PHASE_LANDING  # descente + basse altitude = atterrissage
        return PHASE_DESCENT
    else:
        return PHASE_CRUISE
 
 
def detect_phases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne 'phase' à chaque point du DataFrame.
 
    Nécessite que compute_vario() ait déjà été appelé (colonne vario_fpm).
 
    Valeurs possibles : ground, takeoff, climb, cruise, descent, landing
 
    Un lissage post-détection (mode glissant) est appliqué pour éviter
    les micro-transitions parasites dues au bruit d'altitude.
    """
    if "vario_fpm" not in df.columns:
        raise ValueError("compute_vario() doit être appelé avant detect_phases().")
 
    df = df.copy()
 
    results = []
 
    for flight_id, group in df.groupby("flight_id", sort=False):
        g = group.sort_values("datetime_utc").copy()
 
        # Attribution brute point par point
        g["phase"] = g.apply(_assign_phase, axis=1)
 
        # Lissage : encode en entier → mode glissant → décode en label
        # (rolling().apply() ne fonctionne pas sur des strings)
        phase_order = [PHASE_GROUND, PHASE_TAKEOFF, PHASE_CLIMB,
                       PHASE_CRUISE, PHASE_DESCENT, PHASE_LANDING]
        phase_to_int = {p: i for i, p in enumerate(phase_order)}
        int_to_phase = {i: p for p, i in phase_to_int.items()}
 
        phase_int = g["phase"].map(phase_to_int).astype(float)
        smoothed_int = (
            phase_int
            .rolling(window=PHASE_SMOOTH_WINDOW, center=True, min_periods=1)
            .apply(lambda x: pd.Series(x).mode()[0], raw=True)
            .astype(int)
        )
        g["phase"] = smoothed_int.map(int_to_phase)
 
        results.append(g)
 
    return pd.concat(results).sort_index()
 
 
# ─────────────────────────────────────────────
# 4. ESTIMATION DU VENT
# ─────────────────────────────────────────────
 
def compute_wind(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estime la composante vent à partir de la différence entre
    le cap suivi (heading) et la route effective (track GPS).
 
    Principe : sans vent, heading == track. L'écart révèle la dérive.
    Avec la vitesse propre estimée, on reconstitue le vecteur vent.
 
    Hypothèse clé : speed_kt ≈ vitesse propre (TAS) ce qui est une
    approximation — la vitesse sol (GS) serait plus juste, mais on ne
    l'a pas directement. En pratique l'erreur est faible en croisière.
 
    Trois colonnes ajoutées :
        track_deg      — route effective calculée depuis le GPS (°)
        wind_dir_deg   — direction d'où vient le vent estimé (°)
        wind_speed_kt  — force du vent estimé (kt)
 
    Note : résultats fiables uniquement en croisière stabilisée
    (phase == 'cruise'), sur des segments > 5s sans virage.
    """
    df = df.copy()
 
    results = []
 
    for flight_id, group in df.groupby("flight_id", sort=False):
        g = group.sort_values("datetime_utc").copy()
 
        lats = g["lat"].values
        lons = g["lon"].values
        n = len(g)
 
        # Calcul de la route effective (track) depuis les positions GPS
        tracks = np.full(n, np.nan)
        for i in range(1, n):
            lat1, lon1 = map(radians, [lats[i-1], lons[i-1]])
            lat2, lon2 = map(radians, [lats[i],   lons[i]])
            dlon = lon2 - lon1
            x = sin(dlon) * cos(lat2)
            y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
            track = (degrees(atan2(x, y)) + 360) % 360
            tracks[i] = track
 
        g["track_deg"] = np.round(tracks, 1)
 
        # Composante vent : différence entre vecteur vitesse propre et vecteur sol
        heading_rad = np.radians(g["heading_deg"].values)
        track_rad   = np.radians(g["track_deg"].values)
        speed       = g["speed_kt"].values  # approximation TAS ≈ GS
 
        # Vecteur vitesse propre (cap + vitesse)
        vx_air = speed * np.sin(heading_rad)
        vy_air = speed * np.cos(heading_rad)
 
        # Vecteur vitesse sol (track + vitesse)
        vx_gnd = speed * np.sin(track_rad)
        vy_gnd = speed * np.cos(track_rad)
 
        # Vecteur vent = vitesse sol - vitesse propre
        wx = vx_gnd - vx_air
        wy = vy_gnd - vy_air
 
        wind_speed = np.sqrt(wx**2 + wy**2)
        wind_dir   = (np.degrees(np.arctan2(wx, wy)) + 180 + 360) % 360  # d'où vient le vent
 
        g["wind_dir_deg"]  = np.round(wind_dir, 1)
        g["wind_speed_kt"] = np.round(wind_speed, 1)
 
        # Masquer les estimations non fiables (au sol ou en virage)
        heading_change = g["heading_deg"].diff().abs()
        mask = (g["altitude_ft"] < AIRBORNE_ALT_MIN_FT) | (heading_change > 10)
        g.loc[mask, ["wind_dir_deg", "wind_speed_kt"]] = np.nan
 
        results.append(g)
 
    return pd.concat(results).sort_index()
 
 
# ─────────────────────────────────────────────
# 5. STATISTIQUES PAR VOL
# ─────────────────────────────────────────────
 
def stats_per_flight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule un résumé statistique complet par vol.
 
    Nécessite que enrich() (ou au moins compute_vario + compute_distance
    + detect_phases) ait été appelé.
 
    Retourne un DataFrame avec une ligne par flight_id et les colonnes :
        Temporelles  : date, duration_min, n_points
        Altitude     : alt_max_ft, alt_cruise_ft (médiane en croisière)
        Vitesse      : speed_max_kt, speed_cruise_kt (médiane en croisière)
        Vario        : roc_max_fpm, rod_max_fpm (max montée / descente)
        Distance     : dist_total_nm
        Phases       : time_ground_min, time_climb_min, time_cruise_min, time_descent_min
        Carburant    : fuel_liters (depuis flights_log si dispo)
        Vent         : wind_speed_avg_kt, wind_dir_avg_deg (en croisière)
    """
    rows = []
 
    for flight_id, g in df.groupby("flight_id"):
        g = g.sort_values("datetime_utc")
        duration = (g["datetime_utc"].max() - g["datetime_utc"].min()).total_seconds() / 60
 
        row = {
            "flight_id":    flight_id,
            "date":         g["flight_date"].iloc[0] if "flight_date" in g.columns else None,
            "callsign":     g["callsign"].iloc[0],
            "duration_min": round(duration, 1),
            "n_points":     len(g),
        }
 
        # Altitude
        row["alt_max_ft"] = g["altitude_ft"].max()
 
        # Distance
        if "dist_cum_nm" in g.columns:
            row["dist_total_nm"] = round(g["dist_cum_nm"].max(), 1)
 
        # Vario
        if "vario_fpm" in g.columns:
            row["roc_max_fpm"] = g["vario_fpm"].max(skipna=True)   # montée max
            row["rod_max_fpm"] = g["vario_fpm"].min(skipna=True)   # descente max (négatif)
 
        # Stats par phase
        if "phase" in g.columns:
            dt = g["datetime_utc"].diff().dt.total_seconds().fillna(0)
 
            for phase_label, col_name in [
                (PHASE_GROUND,  "time_ground_min"),
                (PHASE_CLIMB,   "time_climb_min"),
                (PHASE_CRUISE,  "time_cruise_min"),
                (PHASE_DESCENT, "time_descent_min"),
            ]:
                mask = g["phase"] == phase_label
                row[col_name] = round(dt[mask].sum() / 60, 1)
 
            # Altitude et vitesse de croisière (médiane des points en croisière)
            cruise_mask = g["phase"] == PHASE_CRUISE
            if cruise_mask.any():
                row["alt_cruise_ft"]   = g.loc[cruise_mask, "altitude_ft"].median()
                row["speed_cruise_kt"] = g.loc[cruise_mask, "speed_kt"].median()
 
        # Vent moyen en croisière
        if "wind_speed_kt" in g.columns and "phase" in g.columns:
            cruise_mask = g["phase"] == PHASE_CRUISE
            if cruise_mask.any():
                row["wind_speed_avg_kt"] = round(g.loc[cruise_mask, "wind_speed_kt"].median(), 1)
                row["wind_dir_avg_deg"]  = round(g.loc[cruise_mask, "wind_dir_deg"].median(), 1)
 
        # Carburant (depuis flights_log)
        if "fuel_liters" in g.columns:
            fuel = g["fuel_liters"].dropna()
            row["fuel_liters"] = fuel.iloc[0] if not fuel.empty else None
        
        #notes (depuis flights_log)
        if "notes" in g.columns:
            notes_series = g["notes"].dropna()
            row["notes"] = notes_series.iloc[0] if not notes_series.empty else None
            
        # Temps moteur Hobbs (depuis flights_log)
        if "engine_time_h" in g.columns:
            engine_series = g["engine_time_h"].dropna()
            row["engine_time_h"] = engine_series.iloc[0] if not engine_series.empty else None
            
        # --- TDP ---
        if "TDP" in g.columns:
            val = pd.to_numeric(g["TDP"].iloc[0], errors="coerce")
            row["TDP"] = int(val) if pd.notna(val) else 0
        else:
            row["TDP"] = 0
        # -------------------------------------

        # --- is_mine ---
        if "is_mine" in g.columns:
            mine_series = g["is_mine"].dropna()
            # S'il n'y a pas de donnée, on suppose que c'est un vol à toi par défaut (1)
            row["is_mine"] = mine_series.iloc[0] if not mine_series.empty else 1
        # ------------------------------------------

        rows.append(row)
 
    return pd.DataFrame(rows)



def train_fuel_model(df: pd.DataFrame, df_summary: pd.DataFrame, include_others: bool = False, include_tdp: bool = True, selected_aircraft: str = "Tous", excluded_flights: list = None):
    """
    Modèle ultra-robuste avec filtres dynamiques depuis l'interface UI.
    """
    # 1. Calcul du temps de vol réel EN L'AIR d'après le GPS (en Heures)
    phase_times = []
    for fid, g in df.groupby("flight_id"):
        if "phase" not in g.columns:
            continue
        
        # On calcule le delta de temps EXACT en secondes entre chaque point
        dt_sec = g["datetime_utc"].diff().dt.total_seconds().fillna(0)
        
        # On regroupe par phase et on additionne les temps réels
        time_per_phase = dt_sec.groupby(g["phase"]).sum() / 3600.0
        
        t_vol_gps_h = (
            time_per_phase.get("takeoff", 0) + 
            time_per_phase.get("climb", 0) + 
            time_per_phase.get("cruise", 0) + 
            time_per_phase.get("descent", 0) + 
            time_per_phase.get("landing", 0)
        )
        phase_times.append({"flight_id": fid, "T_Vol_GPS_H": t_vol_gps_h})
        
    df_phases = pd.DataFrame(phase_times)
    if df_phases.empty:
        return None, df_summary
        
    # 2. Jointure avec le carnet de vol (ajout de callsign pour le filtre avion)
    for col in ["fuel_liters", "engine_time_h", "is_mine", "callsign", "TDP"]:
        if col not in df_summary.columns:
            df_summary[col] = None

    df_model = pd.merge(df_phases, df_summary[["flight_id", "fuel_liters", "engine_time_h", "is_mine", "callsign", "TDP"]], on="flight_id")
    df_train = df_model.dropna(subset=["fuel_liters", "engine_time_h"])
    
    # 3. FILTRAGE DYNAMIQUE
    # Filtre sur tes vols vs ceux des autres
    if not include_others and "is_mine" in df_train.columns:
        df_train["is_mine"] = df_train["is_mine"].astype(int).astype(bool)
        df_train = df_train[df_train["is_mine"] == True]
        
    # NOUVEAU : Filtre sur les Tours de Piste
    if not include_tdp and "TDP" in df_train.columns:
        df_train = df_train[df_train["TDP"] != 1]

    # Filtre sur l'avion spécifique (BJ, PK, RR...)
    if selected_aircraft != "Tous" and "callsign" in df_train.columns:
        df_train = df_train[df_train["callsign"] == selected_aircraft]
        
    # --- NOUVEAU : Exclusion manuelle des valeurs aberrantes ---
    if excluded_flights:
        df_train = df_train[~df_train["flight_id"].isin(excluded_flights)]
    # -----------------------------------------------------------

    if len(df_train) < 3:
        return None, df_summary

    # 4. LA SOUSTRACTION IMPLACABLE (Hobbs - Vol GPS)
    df_train["T_Sol_Reel_H"] = df_train["engine_time_h"] - df_train["T_Vol_GPS_H"]
    df_train.loc[df_train["T_Sol_Reel_H"] < 0, "T_Sol_Reel_H"] = 0 

    # 5. OPTIMISATION SOUS CONTRAINTE
    X = df_train[["T_Sol_Reel_H", "T_Vol_GPS_H"]].values
    y = df_train["fuel_liters"].values
    
    from scipy.optimize import lsq_linear
    limite_basse = [5.0, 30.0]
    limite_haute = [30.0, 36.0]
    resultat = lsq_linear(X, y, bounds=(limite_basse, limite_haute))
    
    profil_avion = {
        "Sol": resultat.x[0],
        "Vol": resultat.x[1]
    }
    
    # 6. Prédictions pour l'interface
    df_phases_all = pd.merge(df_phases, df_summary[["flight_id", "engine_time_h"]], on="flight_id", how="left")
    df_phases_all["engine_time_h"] = df_phases_all["engine_time_h"].fillna(df_phases_all["T_Vol_GPS_H"] + 0.2)
    df_phases_all["T_Sol_Reel_H"] = df_phases_all["engine_time_h"] - df_phases_all["T_Vol_GPS_H"]
    df_phases_all.loc[df_phases_all["T_Sol_Reel_H"] < 0, "T_Sol_Reel_H"] = 0
    
    X_all = df_phases_all[["T_Sol_Reel_H", "T_Vol_GPS_H"]].values
    df_phases_all["fuel_predicted"] = X_all.dot(resultat.x)
    
    # On ajoute les prédictions ET les temps calculés pour le tableau récapitulatif
    df_summary = pd.merge(df_summary, df_phases_all[["flight_id", "fuel_predicted", "T_Vol_GPS_H", "T_Sol_Reel_H"]], on="flight_id", how="left")
    
    return profil_avion, df_summary
 
 
# ─────────────────────────────────────────────
# 6. PIPELINE COMPLET
# ─────────────────────────────────────────────
 
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet : applique tous les calculs dans le bon ordre.
 
    Ordre obligatoire :
        1. compute_vario()   → nécessaire pour detect_phases()
        2. compute_distance()
        3. detect_phases()   → nécessite vario_fpm
        4. compute_wind()
 
    Retourne le DataFrame enrichi avec toutes les colonnes calculées.
    """
    print("  Calcul du vario...", end=" ")
    df = compute_vario(df)
    print("✓")
 
    print("  Calcul des distances...", end=" ")
    df = compute_distance(df)
    print("✓")
 
    print("  Détection des phases...", end=" ")
    df = detect_phases(df)
    print("✓")
 
    print("  Estimation du vent...", end=" ")
    df = compute_wind(df)
    print("✓")
 
    return df
 
 
# ─────────────────────────────────────────────
# POINT D'ENTRÉE (test rapide)
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.loader import load_all_flights
 
    print("Chargement des vols...")
    df = load_all_flights("data/")
 
    print("\nEnrichissement...")
    df = enrich(df)
 
    print("\n=== Colonnes disponibles ===")
    print(df.columns.tolist())
 
    print("\n=== Résumé par vol ===")
    print(stats_per_flight(df).to_string())
 
    print("\n=== Distribution des phases ===")
    print(df.groupby(["flight_id", "phase"]).size().unstack(fill_value=0))
    
    
    
def compare_fuel(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Isole les vols ayant une consommation réelle et prédite, 
    et calcule l'erreur du modèle.
    """
    df_comp = df_summary.copy()
    
    if "fuel_liters" in df_comp.columns and "fuel_predicted" in df_comp.columns:
        # Erreur : Prédit - Réel 
        # (Positif = on a prévu plus que consommé, Négatif = on a consommé plus que prévu)
        df_comp["fuel_error_L"] = df_comp["fuel_predicted"] - df_comp["fuel_liters"]
    else:
        df_comp["fuel_error_L"] = None
        
    # On ne garde que les lignes où on a pu faire la comparaison
    return df_comp.dropna(subset=["fuel_liters", "fuel_predicted", "fuel_error_L"])