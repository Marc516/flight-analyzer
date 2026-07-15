"""
app.py — Point d'entrée Streamlit pour le Dashboard d'Analyse de Vol.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
from github import Github
import base64

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Flight Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports locaux
from src.loader import load_all_flights
from src.analysis import enrich, stats_per_flight, compare_fuel, train_fuel_model
from src.plots import (
    plot_trajectory,
    plot_trajectory_3d,
    plot_dashboard,
    plot_comparison,
    plot_wind_rose,
    plot_fuel_accuracy
)

# ─────────────────────────────────────────────
# 1. CHARGEMENT ET CACHE DES DONNÉES
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Chargement des vols en cours...")
def get_data(data_dir: str = "data"):
    # Charge les CSV/KML et le flights_log.csv
    df_raw = load_all_flights(data_dir)
    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Enrichissement (vitesses, vario, phases, vent)
    df = enrich(df_raw)
    
    # Résumé par vol
    df_summary = stats_per_flight(df)
    return df, df_summary

# Chargement
df, df_summary = get_data()


# ─────────────────────────────────────────────
# 1.5 SÉPARATION DES DONNÉES (UI vs MODÈLE)
# ─────────────────────────────────────────────

# Safety net: If the column doesn't exist yet, assume all flights are yours
if "is_mine" not in df_summary.columns:
    df_summary["is_mine"] = True
else:
    # Ensure it's treated as a boolean (handles True/False, 1/0, or empty)
    df_summary["is_mine"] = df_summary["is_mine"].fillna(1).astype(int).astype(bool)

# Create a filtered summary strictly for your UI/Dashboard
df_personal_ui = df_summary[df_summary["is_mine"] == True]


# ─────────────────────────────────────────────
# 2. GESTION DE LA NAVIGATION (STATE)
# ─────────────────────────────────────────────

# On garde en mémoire la page actuelle ("home", "fuel" ou un "flight_id")
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

def set_page(page: str):
    st.session_state.current_page = page

# ─────────────────────────────────────────────
# 3. BARRE LATÉRALE (NAVIGATION & IMPORT)
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("🛩️ Flight Analyzer")
    
    if st.button("🏠 **Vue d'ensemble**", key="btn_home"):
        set_page("home")
        
    if st.button("⛽ **Prédiction Carburant**", key="btn_fuel"):
        set_page("fuel")
        
    st.markdown("---")
    st.markdown("### 📂 Vos Vols")

    current = st.session_state.current_page
    
    if df_personal_ui.empty:
        st.info("Aucun vol trouvé dans le dossier `data/`.")
    else:
        # Tri des vols du plus récent au plus ancien
        df_vols_tries = df_personal_ui.sort_values(by="date", ascending=False)
        
        # ── Affichage de la liste des vols ──
        # ── Affichage de la liste des vols ──
        for _, row in df_vols_tries.iterrows():
            vol_id = row["flight_id"]
            date_vol = row.get("date", "Date inconnue")
            note = row.get("notes", "")

            # 1. Base du titre du bouton avec le bon doigt/avion
            if current == vol_id:
                titre_bouton = f"👉 {date_vol} \n"
            else:
                titre_bouton = f"✈️ {date_vol} \n"

            # 2. Ajout de la note DANS le bouton (sur une 2ème ligne avec \n), sans les astérisques
            if pd.notna(note) and note != "" and note != "Aucune note":
                titre_bouton += f"\n{note}"

            # 3. Création du bouton
            if st.button(titre_bouton, key=f"btn_{vol_id}", use_container_width=True):
                set_page(vol_id)
                st.rerun()

    # ── 📥 ASSISTANT D'IMPORTATION (V2) ──
    st.markdown("---")
    st.markdown("### 📥 Assistant d'importation")
    
    uploaded_files = st.file_uploader(
        "Déposez vos fichiers (.csv et .kml)", 
        type=["csv", "kml"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        # On cherche le CSV pour pré-remplir les infos
        csv_file = next((f for f in uploaded_files if f.name.endswith('.csv')), None)
        
        if csv_file:
            # 1. Extraction auto pour l'ID unique
            df_temp = pd.read_csv(csv_file, nrows=1)
            callsign = str(df_temp["Callsign"].iloc[0]) if "Callsign" in df_temp.columns else "FXXXX"
            dt_start = pd.to_datetime(df_temp["UTC"].iloc[0]) if "UTC" in df_temp.columns else pd.Timestamp.now()
            
            # Format d'ID : YYYY-MM-DD_HHMM_CALLSIGN
            generated_id = f"{dt_start.strftime('%Y-%m-%d_%H%M')}_{callsign}"
            
            st.info(f"Identifiant généré : `{generated_id}`")
            
            # 2. Formulaire manuel
            with st.form("form_import_vol"):
                col1, col2 = st.columns(2)
                dep = col1.text_input("Aérodrome Départ", value="LFCL")
                arr = col2.text_input("Aérodrome Arrivée", value="LFCL")
                
                col3, col4 = st.columns(2)
                fuel = col3.number_input("Carburant (L)", min_value=0.0, step=0.1, format="%.1f")
                engine_time = col4.number_input("Compteur Hobbs (Heures déc.)", min_value=0.0, step=0.1, format="%.1f", help="Ex: 1.2 pour 1h12. Laissez 0.0 si inconnu.")
                
                # --- NOUVEAU : Les cases à cocher côte à côte ---
                col5, col6 = st.columns(2)
                with col5:
                    is_mine = st.checkbox("C'est mon vol (is_mine)", value=True)
                with col6:
                    is_tdp = st.checkbox("Tour de Piste (TDP)", value=False)
                # -------------------------------------------------
                
                notes = st.text_area("Notes", placeholder="Ex: Nav Pamiers et retour")
                
                submit = st.form_submit_button("Enregistrer le vol")
                
                if submit:
                    with st.spinner("☁️ Envoi vers GitHub en cours... ne quittez pas la page."):
                        try:
                            # 1. Connexion au Coffre-fort GitHub
                            g = Github(st.secrets["GITHUB_TOKEN"])
                            repo = g.get_repo(st.secrets["GITHUB_REPO"])
                            
                            # 2. Envoi des fichiers bruts (CSV et KML)
                            for file_to_save in uploaded_files:
                                extension = Path(file_to_save.name).suffix
                                file_path = f"data/raw/{generated_id}{extension}"
                                
                                # Création physique du fichier sur GitHub
                                repo.create_file(
                                    path=file_path, 
                                    message=f"Ajout trace {generated_id}", 
                                    content=file_to_save.getvalue(), 
                                    branch="main"
                                )
                            
                            # 3. Préparation de la nouvelle ligne du carnet
                            fuel_to_save = fuel if fuel > 0.0 else ""
                            engine_to_save = engine_time if engine_time > 0.0 else ""
                            
                            new_line = pd.DataFrame([{
                                "date": dt_start.strftime('%Y-%m-%d'),
                                "flight_id": generated_id,
                                "fuel_liters": fuel_to_save,
                                "engine_time_h": engine_to_save,
                                "departure": dep.upper(),
                                "arrival": arr.upper(),
                                "is_mine": 1 if is_mine else 0,
                                "TDP": 1 if is_tdp else 0,
                                "notes": notes
                            }])
                            
                            # Convertir en texte CSV
                            new_csv_string = new_line.to_csv(header=False, index=False)
                            
                            # 4. Téléchargement, modification et écrasement du flights_log.csv
                            log_path = "data/flights_log.csv"
                            file_contents = repo.get_contents(log_path, ref="main")
                            
                            # Décoder le texte existant
                            existing_content = base64.b64decode(file_contents.content).decode("utf-8")
                            
                            # Sécurité : ajouter un saut de ligne si le fichier n'en a pas à la fin
                            if not existing_content.endswith('\n'):
                                existing_content += '\n'
                                
                            updated_content = existing_content + new_csv_string
                            
                            # Envoyer la version mise à jour
                            repo.update_file(
                                path=file_contents.path, 
                                message=f"Mise à jour log {generated_id}", 
                                content=updated_content, 
                                sha=file_contents.sha, 
                                branch="main"
                            )
                            
                            st.success("✅ Vol enregistré sur le Cloud et synchronisé avec succès !")
                            st.cache_data.clear()
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la synchronisation : {e}")
                            st.info("Vérifiez que vos 'Secrets' Streamlit sont bien configurés et corrects.")
        else:
            st.warning("⚠️ Glissez au moins le fichier .csv pour extraire les infos de base.")
                
     # ── Bouton de rafraîchissement manuel du cache ──
    st.markdown("---")
    if st.button("🔄 Rafraîchir les données", use_container_width=True, help="Vide le cache et recharge les fichiers CSV/KML et notes"):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────
# 4. AFFICHAGE PRINCIPAL (CONTENU)
# ─────────────────────────────────────────────

current = st.session_state.current_page

if df.empty:
    st.warning("⚠️ Aucune donnée n'a été chargée. Placez vos fichiers KML/CSV dans le dossier `data/`.")

elif current == "home":
    st.title("🏠 Vue d'ensemble des vols")
    st.markdown("Visualisation combinée de l'ensemble de votre carnet de vol.")
    
    tab_map, tab_comp, tab_table = st.tabs(["🗺️ Cartographie Globale", "📊 Comparatif", "📝 Table de Synthèse"])
    
    with tab_map:
        try:
            # --- CORRECTION : Filtrer les points GPS sur tes propres vols ---
            vols_perso_ids = df_personal_ui["flight_id"].unique()
            df_carte_perso = df[df["flight_id"].isin(vols_perso_ids)]
            
            # On passe le DataFrame filtré à la fonction de tracé
            m = plot_trajectory(df_carte_perso, save=False)
            # ----------------------------------------------------------------
            
            components.html(m._repr_html_(), height=600)
        except Exception as e:
            st.error(f"Erreur d'affichage de la carte globale : {e}")
            
    with tab_comp:
        try:
            fig_comp = plot_comparison(df_personal_ui, save=False)
            st.plotly_chart(fig_comp, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur d'affichage du comparatif : {e}")
            
    with tab_table:
        st.dataframe(df_personal_ui, use_container_width=True, hide_index=True)

elif current == "fuel":
    st.title("⛽ Modèle Carburant (Hobbs vs GPS)")
    
    # --- NOUVEAU BLOC : PANNEAU DE CONTRÔLE ---
    st.markdown("### ⚙️ Base de données d'apprentissage")
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        # Checkbox pour inclure/exclure
        include_others = st.checkbox("Inclure les vols importés (is_mine = 0)", value=False)
        include_tdp = st.checkbox("Inclure les Tours de Piste (TDP = 1)", value=True)
        
    with col_f2:
        # Construction dynamique de la liste des avions disponibles
        avions_dispos = ["Tous"]
        if "callsign" in df_summary.columns:
            liste_avions = sorted(df_summary["callsign"].dropna().unique().tolist())
            avions_dispos.extend(liste_avions)
            
        selected_aircraft = st.selectbox("Filtrer par avion spécifique", options=avions_dispos)
    
    # On liste uniquement les vols qui ont une donnée de carburant
    vols_avec_carburant = df_summary.dropna(subset=["fuel_liters"])["flight_id"].tolist()
    vols_exclus = st.multiselect("❌ Exclure manuellement certains vols (ex: valeurs aberrantes) :", options=vols_avec_carburant)
    # -----------------------------------
    
    st.markdown("---")
    # On passe le nouveau paramètre vols_exclus à l'IA
    profil_avion, df_summary_with_preds = train_fuel_model(df, df_summary, include_others, include_tdp, selected_aircraft, excluded_flights=vols_exclus)
    
    st.markdown("---")
    # ------------------------------------------
    
    # On passe les nouveaux paramètres à notre fonction IA
    profil_avion, df_summary_with_preds = train_fuel_model(df, df_summary, include_others, include_tdp, selected_aircraft)
    
    if profil_avion is None:
        st.warning("⚠️ L'algorithme a besoin d'au moins 3 vols correspondant à ces filtres pour s'entraîner.")
    else:
        st.success(f"✅ Modèle entraîné sur la configuration sélectionnée !")
        
        tab_simu, tab_perf = st.tabs(["🧮 Simulateur de Nav", "📈 Performance du Modèle"])
        
        with tab_simu:
            # --------------------------------------------------------
            st.markdown("### Profil de Consommation Détecté")
            col1, col2 = st.columns(2)
            col1.metric("Roulage / Sol", f"{profil_avion['Sol']:.1f} L/h")
            col2.metric("En Vol", f"{profil_avion['Vol']:.1f} L/h")
            # ---------------------------------------------------------
            
            st.markdown("---")
            st.markdown("### 🧮 Préparer un vol")
            
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                sim_vol = st.number_input("Temps de vol estimé (minutes)", min_value=10, value=60, step=5)
            with col_in2:
                sim_sol = st.number_input("Temps de roulage prévu (minutes)", min_value=5, value=15, step=5)
            
            # Application des taux (conversion L/h en L/min)
            conso_sol = sim_sol * (profil_avion['Sol'] / 60)
            conso_vol = sim_vol * (profil_avion['Vol'] / 60)
            
            total_bloc = conso_sol + conso_vol
            reserve_vfr = 30 * (profil_avion['Vol'] / 60)
            carburant_min = total_bloc + reserve_vfr
            
            st.markdown("#### Résultat de la simulation")
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Délestage (Vol)", f"{conso_vol:.1f} L")
            col_r2.metric("Bloc à Bloc", f"{total_bloc:.1f} L")
            col_r3.metric("Min. Réglementaire", f"{carburant_min:.1f} L", "Inclut 30 min réserve VFR")

        with tab_perf:
            df_filtered_perf = df_summary_with_preds.copy()
            
            # Application des filtres pour l'affichage
            if not include_others and "is_mine" in df_filtered_perf.columns:
                df_filtered_perf["is_mine"] = df_filtered_perf["is_mine"].fillna(1).astype(int).astype(bool)
                df_filtered_perf = df_filtered_perf[df_filtered_perf["is_mine"] == True]
                
            # NOUVEAU : Application du filtre visuel pour le TDP
            if not include_tdp and "TDP" in df_filtered_perf.columns:
                df_filtered_perf = df_filtered_perf[df_filtered_perf["TDP"] != 1]
                
            if selected_aircraft != "Tous" and "callsign" in df_filtered_perf.columns:
                df_filtered_perf = df_filtered_perf[df_filtered_perf["callsign"] == selected_aircraft]
                
            # --- Appliquer l'exclusion visuelle ---
            if vols_exclus:
                df_filtered_perf = df_filtered_perf[~df_filtered_perf["flight_id"].isin(vols_exclus)]
            
            # 3. On génère les métriques et le graphique UNIQUEMENT sur ces vols filtrés
            try:
                df_comp = compare_fuel(df_filtered_perf)
                
                # Vérification sécurité s'il reste des vols à comparer
                if not df_comp.empty:
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.metric("Erreur Moyenne Absolue (MAE)", f"{df_comp['fuel_error_L'].abs().mean():.1f} L")
                    with col_p2:
                        st.metric("Biais Moyen", f"{df_comp['fuel_error_L'].mean():.1f} L")
                        
                    fig_fuel = plot_fuel_accuracy(df_comp, save=False)
                    if fig_fuel:
                        st.plotly_chart(fig_fuel, use_container_width=True)
                    
                    # --- TABLEAU RÉCAPITULATIF DES VOLS ---
                    st.markdown("---")
                    st.markdown("### 📋 Détail des vols utilisés pour l'entraînement")
                    
                    # On copie uniquement les colonnes intéressantes pour ne pas polluer l'écran
                    df_table = df_comp[["flight_id", "callsign", "engine_time_h", "T_Vol_GPS_H", "T_Sol_Reel_H", "fuel_liters"]].copy()
                    
                    # Renommage propre des colonnes pour les humains
                    df_table = df_table.rename(columns={
                        "flight_id": "ID Vol",
                        "callsign": "Appareil",
                        "engine_time_h": "Hobbs (h)",
                        "T_Vol_GPS_H": "Vol GPS (h)",
                        "T_Sol_Reel_H": "Roulage (h)",
                        "fuel_liters": "Carburant réel (L)"
                    })
                    
                    # Arrondir les calculs GPS à 2 décimales
                    df_table["Vol GPS (h)"] = df_table["Vol GPS (h)"].round(2)
                    df_table["Roulage (h)"] = df_table["Roulage (h)"].round(2)
                    
                    st.dataframe(df_table, use_container_width=True, hide_index=True)
                    # ------------------------------------------------
                else:
                    st.info("Aucune donnée de carburant réel n'est disponible pour cette sélection.")
                    
            except Exception as e:
                st.error(f"Erreur d'affichage du graphique de performance : {e}")

else:
    # ── Analyse détaillée d'un vol ──
    vol_choisi = current
    
    info_vol = df_summary[df_summary["flight_id"] == vol_choisi].iloc[0]
    date_vol = info_vol.get("date", "Date inconnue")
    note_vol = info_vol.get("notes", "")
    
    # Titre propre sans astérisques
    titre = f"Vol du {date_vol}"
    if pd.notna(note_vol) and note_vol and note_vol != "Aucune note":
        titre += f" — {note_vol}"
        
    st.title(titre)
    st.caption(f"ID du vol : `{vol_choisi}`")
    
    # ── Les 4 Onglets ──
    tab_dash, tab_map, tab_map3d, tab_wind = st.tabs([
        "📊 Dashboard Complet", 
        "🗺️ Carte 2D", 
        "🌍 Carte 3D", 
        "💨 Analyse du Vent"
    ])
    
    with tab_dash:
        try:
            fig_dash = plot_dashboard(df, vol_choisi, save=False)
            st.plotly_chart(fig_dash, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur (Dashboard) : {e}")
            
    with tab_map:
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            color_opt = st.radio("Coloration :", ["phase", "altitude", "speed"], horizontal=True, key=f"radio_{vol_choisi}")
        with col_opt2:
            import os
            est_en_local = not os.getcwd().startswith("/mount")
            
            options_cartes = [
                "Carte Relief VFR (Stamen Terrain)", 
                "Esri Satellite (Vue Réelle)", 
                "OpenTopoMap (Topographique)", 
                "OpenStreetMap (Standard)"
            ]
            
            map_style = st.selectbox(
                "Fond de carte :", 
                options_cartes,
                # Index 0 (Stamen) chez toi, Index 1 (Satellite) sur le cloud
                index=0 if est_en_local else 1,
                key=f"map_{vol_choisi}"
            )
        
        # Clé OpenAIP — à stocker dans st.secrets ou saisie manuelle
        openaip_key = st.secrets.get("OPENAIP_KEY", None)  # méthode propre
        # ou en saisie manuelle pour tester :
        # openaip_key = st.text_input("Clé OpenAIP (optionnel)", type="password")

        try:
            m = plot_trajectory(
                df,
                flight_id=vol_choisi,
             color_by=color_opt,
                map_style=map_style,
                openaip_key=openaip_key,   # ← la clé ici
                save=False
            )
            components.html(m._repr_html_(), height=650)
        except Exception as e:
            st.error(f"Erreur (Carte 2D) : {e}")
            
    with tab_map3d:
        st.info("💡 **Astuce :** Maintenez le bouton droit de la souris enfoncé (ou Maj + clic gauche) et bougez la souris pour faire pivoter la carte en 3D !")
        try:
            deck_3d = plot_trajectory_3d(df, flight_id=vol_choisi)
            st.pydeck_chart(deck_3d, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur (Carte 3D) : {e}")
            
    with tab_wind:
        col_text, col_plot = st.columns([1, 2])
        with col_text:
            st.info("Rose des vents estimée en croisière (basée sur la dérive de la trace GPS par rapport au cap).")
        with col_plot:
            try:
                fig_wind = plot_wind_rose(df, flight_id=vol_choisi, save=False)
                if fig_wind:
                    st.pyplot(fig_wind)
                else:
                    st.warning("Données insuffisantes pour estimer le vent.")
            except Exception as e:
                st.error(f"Erreur (Vent) : {e}")