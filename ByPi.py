import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots


# Configurer la mise en page pour utiliser toute la largeur disponible
st.set_page_config(layout="wide")
# Configuration du thème seaborn
sns.set_theme(style="whitegrid")
# Initialisation des variables globales pour les dates
cal_debut = None
cal_fin = None

# Variables globales
df_energie = None  # Pour les données énergétiques
colonne_p10 = None 

# Interface Streamlit
st.title("Analyse de Consommation Énergétique")
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier Excel pour les P10", type=["xlsx"], key="file_uploader")
uploaded_file_temperature = st.sidebar.file_uploader("Choisissez un fichier Excel pour les températures", type=["xlsx"], key="file_uploader_temperature")

# Sélectionner le mode d'analyse
mode = st.sidebar.selectbox("Choisissez un mode d'analyse :", ["Comparaison par période", "Analyse et rapport", "Export des données"])

def graphe_by_analyse_heatmap():
    global colonne_p10
    all_data = analyse()

    # Vérifier si all_data n'est pas None
    if all_data is not None and not all_data.empty:
        # Créer un DataFrame pivot pour structurer les données pour la heatmap
        periodes = all_data['Période'].unique()
        heatmap_data = pd.DataFrame()

        for periode in periodes:
            data_periode = all_data[(all_data['Période'] == periode) & (all_data[colonne_p10] > 0)]
            data_periode = data_periode.set_index('Date')
            data_periode.index = pd.to_datetime(data_periode.index)

            # Resample pour une agrégation par intervalles de 10 minutes
            data_periode_resampled = data_periode[colonne_p10].resample("10T").mean()
            
            # Ajouter les données dans la structure du DataFrame pour la heatmap
            heatmap_data[periode] = data_periode_resampled

        # Remplir les valeurs manquantes avec 0 (ou tout autre valeur par défaut)
        heatmap_data.fillna(0, inplace=True)

        # Créer la heatmap
        fig_heatmap = go.Figure(
            go.Heatmap(
                z=heatmap_data.values,                   # Valeurs de consommation
                x=heatmap_data.index,                    # Dates en tant qu'axe x
                y=heatmap_data.columns,                  # Périodes en tant qu'axe y
                colorscale="Viridis",                    # Palette de couleurs
                colorbar=dict(title="Consommation (kWh)")  # Titre pour l'échelle de couleurs
            )
        )

        # Personnalisation des titres et des polices
        fig_heatmap.update_layout(
            title='Carte thermique des consommations par période',
            xaxis_title='Date',
            yaxis_title='Période de fonctionnement',
            title_font=dict(family="Times New Roman", size=24),
            xaxis_title_font=dict(family="Times New Roman", size=20),
            yaxis_title_font=dict(family="Times New Roman", size=20),
            font=dict(family="Times New Roman", size=14)
        )

        # Afficher la heatmap dans Streamlit
        st.plotly_chart(fig_heatmap)
    else:
        st.error("Aucune donnée à afficher.")

def graphe_regression_tempo():
    global colonne_p10, colonne_temperature, df_temperature, cal_debut, cal_fin
    all_data = analyse()

    # Vérifiez que df_temperature existe et a des valeurs valides
    if df_temperature is not None and not df_temperature.empty:
        df_temperature_clean = df_temperature.dropna(subset=[colonne_temperature])
        df_temperature_resampled = df_temperature_clean.resample("1h").mean()  # Resample par heure

    if all_data is not None and not all_data.empty:
        fig_histogram = go.Figure()
        periodes = all_data['Période'].unique()
        couleurs_periodes = ["Blue", "MediumSeaGreen", "Red"]

        # Tracer l'histogramme des consommations
        for idx, periode in enumerate(periodes):
            data_periode = all_data[(all_data['Période'] == periode) & (all_data[colonne_p10] > 0)]
            data_periode = data_periode.set_index('Date')
            data_periode.index = pd.to_datetime(data_periode.index)
            data_periode_resampled = data_periode[colonne_p10].resample("10min").mean()

            # Ajouter les barres de consommation
            fig_histogram.add_trace(
                go.Bar(
                    x=data_periode_resampled.index,
                    y=data_periode_resampled,
                    name=f'{periode}',
                    marker=dict(color=couleurs_periodes[idx % len(couleurs_periodes)]),
                    opacity=0.8,
                    yaxis="y1"  # Associe cette trace à l'axe Y principal (y1)
                )
            )

            # Filtrer df_temperature pour obtenir les valeurs correspondant aux index de la période
            temperature_periode = df_temperature_resampled[df_temperature_resampled.index.isin(data_periode_resampled.index)]

            # Ajouter la courbe de température pour la période correspondante avec un axe Y secondaire
            fig_histogram.add_trace(
                go.Scatter(
                    x=temperature_periode.index,
                    y=temperature_periode[colonne_temperature],
                    mode='lines',
                    name=f'Température',
                    line=dict(color=couleurs_periodes[idx % len(couleurs_periodes)]),
                    yaxis="y2"  # Associe cette trace à l'axe Y secondaire (y2)
                )
            )

        # Mise en page avec un second axe Y pour les températures
        fig_histogram.update_layout(
            title=f'Consommations électriques et Température : {colonne_p10}',
            xaxis_title='Date',
            yaxis=dict(
                title='Consommations en kWh',
                titlefont=dict(family="Times New Roman", size=20),
                tickfont=dict(family="Times New Roman", size=18),
            ),
            yaxis2=dict(
                title='Température (°C)',
                titlefont=dict(family="Times New Roman", size=20),
                tickfont=dict(family="Times New Roman", size=18),
                overlaying='y',  # Superpose l'axe secondaire sur l'axe Y principal
                side='right'  # Place l'axe secondaire à droite
            ),
            title_font=dict(family="Times New Roman", size=24),
            xaxis_title_font=dict(family="Times New Roman", size=24),
            font=dict(family="Times New Roman", size=16)
        )

        st.plotly_chart(fig_histogram)
    else:
        st.error("Aucune donnée à afficher.")

# Fonction pour afficher les graphiques pour chaque période en regrresion linèaire avec les températures
def graphe_regression_lineaire():
    global colonne_p10, colonne_temperature, df_temperature, cal_debut, cal_fin
    all_data = analyse()
    #température de consignes pour les périodes

    # Vérifiez que df_temperature existe et a des valeurs valides
    if df_temperature is not None and not df_temperature.empty:
        df_temperature_clean = df_temperature.dropna(subset=[colonne_temperature])
        df_temperature_resampled = df_temperature_clean.resample("1h").mean()  # Resample par heure
    else:
        st.error("Données de température manquantes ou invalides.")
        return

    if all_data is not None and not all_data.empty:
        periodes = all_data['Période'].unique()
        dfs_periode = []
        couleurs_periodes = ["Blue", "MediumSeaGreen", "Red"]

        # Préparation des données de chaque période
        for idx, periode in enumerate(periodes):
            data_periode = all_data[(all_data['Période'] == periode) & (all_data[colonne_p10] > 0)]
            data_periode = data_periode.set_index('Date')
            data_periode.index = pd.to_datetime(data_periode.index)
            data_periode_resampled = data_periode[colonne_p10].resample("1h").mean()

            # Filtrer df_temperature pour obtenir les valeurs correspondant aux index de la période
            temperature_periode = df_temperature_resampled[df_temperature_resampled.index.isin(data_periode_resampled.index)]

            # Fusion des consommations et températures dans un seul DataFrame pour cette période
            df_merged = pd.DataFrame({
                'Consommation': data_periode_resampled,
                'Température': temperature_periode[colonne_temperature]
            }).dropna()  # Supprimer les lignes avec des valeurs NaN
            
            df_merged['Période'] = periode
            df_merged['Couleur'] = couleurs_periodes[idx]
            df_merged['Date'] = df_merged.index

            dfs_periode.append(df_merged)

        # Concaténation des données de toutes les périodes
        df_final = pd.concat(dfs_periode)

        # Initialisation de la figure
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=["<b>Regression des consommations électriques en rapport avec les tempratures<br>Selon les périodes de fonctionnement du site"],
        )

        # Ajouter des traces pour chaque période
        for idx, periode in enumerate(periodes):
            key_checkbox = f"afficher_annotations_{periode}"  # Clé unique pour chaque période
            afficher_annotations = st.checkbox(f"Afficher les annotations pour {periode}", value=True, key=key_checkbox)

            data = df_final[df_final['Période'] == periode]
            if data.empty:
                st.warning(f"Aucune donnée disponible pour la période {periode}.")
                continue

            # Ajouter un scatter pour la période
            fig.add_trace(
                go.Scatter(
                    x=data[f'Consommation'],
                    y=data['Température'],
                    mode='markers',
                    name=f"{periode}",
                    marker=dict(color=data['Couleur'].iloc[0], size=data['Consommation'] / data['Consommation'].max() * 20),
                    hovertemplate="<b>Date:</b> %{text}<br><b>Consommation en kWh:</b> %{x}<br><b>Température en °C:</b> %{y}<br><extra></extra>",
                    text=data['Date'].dt.strftime('%Y-%m-%d %H:%M')
                ),
                row=1, col=1
            )

            # Calcul et ajout de la droite de régression linéaire
            if not data.empty:
                slope, intercept = np.polyfit(data['Consommation'], data['Température'], 1)
                regression_line = slope * data['Consommation'] + intercept
                r_value = np.corrcoef(data['Consommation'], data['Température'])[0, 1]

                fig.add_trace(
                    go.Scatter(
                        x=data['Consommation'],
                        y=regression_line,
                        mode='lines',
                        name=f"Tendance {periode} (R={r_value:.2f})",
                        line=dict(color=data['Couleur'].iloc[0], dash='dot')
                    ),
                    row=1, col=1
                )

            # Ajouter des annotations si la checkbox est activée
            if afficher_annotations:
                # Annotations pour les quantiles
                quantiles = data['Consommation'].quantile([0.25, 0.5, 0.75]).to_list()
                quantiles_percentages = [25, 50, 75]
                for quantile, percentage in zip(quantiles, quantiles_percentages):
                    closest_point = data.iloc[(data['Consommation'] - quantile).abs().argsort()[:1]]
                    consommation = closest_point['Consommation'].values[0]
                    temperature = closest_point['Température'].values[0]
                    date = closest_point['Date'].dt.strftime('%Y-%m-%d %H:%M').values[0]

                    fig.add_annotation(
                        x=consommation,
                        y=temperature,
                        text=f"{date}, {temperature:.2f} °C<br>{percentage}% des P10 sont < à {consommation:.2f} kWh",
                        showarrow=True,
                        arrowhead=2,
                        ax=40,
                        ay=-40,
                        font=dict(size=12, color="black"),
                        bgcolor="lightyellow",
                        bordercolor="black",
                        borderwidth=1
                    )

                # Annotations pour Température max et min
                max_temp = data.loc[data['Température'].idxmax()]
                min_temp = data.loc[data['Température'].idxmin()]

                for temp_data, label in zip([max_temp, min_temp], ["Max", "Min"]):
                    fig.add_annotation(
                        x=temp_data['Consommation'],
                        y=temp_data['Température'],
                        text=(
                            f"T° {label} en {periode}<br>{temp_data['Date'].strftime('%Y-%m-%d %H:%M')}<br>"
                            f"({temp_data['Consommation']:.2f} kWh, {temp_data['Température']:.2f} °C)"
                        ),
                        showarrow=True,
                        arrowhead=2,
                        ax=40 if label == "Max" else -40,
                        ay=-40,
                        font=dict(size=14, color="black"),
                        bgcolor="lightpink" if label == "Max" else "lightblue",
                        bordercolor="black",
                        borderwidth=1
                    )

        # Mise en forme du graphique
        try:
            fig.update_layout(
                height=800,
                title="<b>Analyse de la Consommation en corrélation avec des Températures",
                title_font=dict(family="Times New Roman", size=24),
                font=dict(family="Times New Roman", size=18),
                editable=True
            )
        except Exception as e:
            print(f"Erreur lors de la mise à jour de la mise en page: {e}")

        fig.update_xaxes(title_text="<b>Consommations électriques mesurées en (kWh)", row=1, col=1)
        fig.update_yaxes(title_text="<b>Température mesurées en (°C)", row=1, col=1)

        # Afficher le graphique
        st.plotly_chart(fig)
    else:
        st.error("Aucune donnée à afficher.")

def graphe_tempo():
    global colonne_p10, colonne_temperature, df_temperature, cal_debut, cal_fin
    all_data = analyse()

    # Vérifiez que df_temperature existe et a des valeurs valides
    if df_temperature is not None and not df_temperature.empty:
        df_temperature_clean = df_temperature.dropna(subset=[colonne_temperature])
        df_temperature_resampled = df_temperature_clean.resample("1h").mean()  # Resample par heure

    if all_data is not None and not all_data.empty:
        periodes = all_data['Période'].unique()
        dfs_periode = []
        couleurs_periodes = ["Yellow", "MediumSeaGreen", "Red", "Blue"]

        # Préparation des données de chaque période
        for periode in periodes:
            data_periode = all_data[(all_data['Période'] == periode) & (all_data[colonne_p10] > 0)]
            data_periode = data_periode.set_index('Date')
            data_periode.index = pd.to_datetime(data_periode.index)
            data_periode_resampled = data_periode[colonne_p10].resample("1h").mean()

            # Filtrer df_temperature pour obtenir les valeurs correspondant aux index de la période
            temperature_periode = df_temperature_resampled[df_temperature_resampled.index.isin(data_periode_resampled.index)]
            
            # Fusion des consommations et températures dans un seul DataFrame pour cette période
            df_merged = pd.DataFrame({
                'Consommation': data_periode_resampled,
                'Température': temperature_periode[colonne_temperature]
            }).dropna()  # Supprimer les lignes avec des valeurs NaN
            df_merged['Période'] = periode

            dfs_periode.append(df_merged)

        # Concaténation des données de toutes les périodes
        df_final = pd.concat(dfs_periode)

        # Création d'un subplot avec une colonne et un nombre de lignes égal au nombre de périodes
        fig = make_subplots(
            rows=len(periodes), cols=1,
            subplot_titles=[f"Période: {periode}" for periode in periodes],
            vertical_spacing=0.1
        )

        # Ajout de chaque période dans un sous-graphe
        for idx, periode in enumerate(periodes):
            data = df_final[df_final['Période'] == periode]
            fig.add_trace(
                go.Scatter(
                    x=data['Consommation'],
                    y=data['Température'],
                    mode='markers',
                    name=f"Consommation vs Température - {periode}",
                    marker=dict(color=couleurs_periodes[idx], size=15),  # Utilisation de la couleur pour chaque période
                ),
                row=idx + 1, col=1
            )

            # Ajout de la tendance linéaire pour chaque période
            if not data.empty:
                trendline = px.scatter(data, x='Consommation', y='Température', trendline="ols")
                trendline_data = trendline.data[1]  # La tendance est la deuxième trace
                fig.add_trace(trendline_data, row=idx + 1, col=1)

        # Personnalisation du layout
        fig.update_layout(
            height=500 * len(periodes),  # Hauteur ajustable pour chaque période
            title="Relation entre Consommation électrique et Température par Période",
            title_font=dict(family="Times New Roman", size=24),
            xaxis_title="Température (°C)",
            yaxis_title="Consommation (kWh)",
            font=dict(family="Times New Roman", size=14)
        )

        # Ajustement des titres des axes pour chaque sous-graphe
        fig.update_xaxes(title_text="Consommation (kWh)")
        fig.update_yaxes(title_text="Température (°C)")

        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig)
    else:
        st.error("Aucune donnée à afficher.")

def by_analyse():
    global colonne_p10, data_periode
    
    cal_debut_dt = pd.to_datetime(cal_debut)
    cal_fin_dt = pd.to_datetime(cal_fin)
    
    all_data = analyse()
    if all_data is None or all_data.empty:
        st.error("Aucune donnée disponible.")
        return
    
    data_periode = all_data
    if not pd.api.types.is_datetime64_any_dtype(data_periode["Date"]):
        data_periode["Date"] = pd.to_datetime(data_periode["Date"])
    
    data_mois_actuel = data_periode[
        (data_periode["Date"] >= cal_debut_dt) & (data_periode["Date"] <= cal_fin_dt)
    ]
    
    consommation_actuelle = data_mois_actuel[colonne_p10].sum()
    st.write(f"### ⚡ Energie totale mesurée **{consommation_actuelle:.0f} kWh**, entre {cal_debut_dt.date()} & {cal_fin_dt.date()} du point de mesure : {colonne_p10}")
    
    if "Période" in data_periode.columns:
        periodes_uniques = data_periode["Période"].unique()
        colonnes = st.columns(len(periodes_uniques))
        
        efficiences = []
        for periode in periodes_uniques:
            data_période_actuel = data_mois_actuel[data_mois_actuel["Période"] == periode]
            somme_actuelle = data_période_actuel[colonne_p10].sum()
            duree = 14 if "08h-22h" in periode else 3 if "05h-08h" in periode else 7 if "22h-05h" in periode else 1
            efficience_periode = somme_actuelle / duree if duree else 0  
            efficiences.append(efficience_periode)
        
        for i, periode in enumerate(periodes_uniques):
            with colonnes[i]:
                data_période_actuel = data_mois_actuel[data_mois_actuel["Période"] == periode]
                somme_actuelle = data_période_actuel[colonne_p10].sum()
                duree = 14 if "08h-22h" in periode else 3 if "05h-08h" in periode else 7 if "22h-05h" in periode else 1
                efficience_periode = somme_actuelle / duree if duree else 0  
                couleur = get_color(efficience_periode, efficiences)
                
                st.markdown(f"<h4 style='color:{couleur};'> {periode}</h4>", unsafe_allow_html=True)
                st.write(f"###### ⚡ Consommation de la période : **{somme_actuelle:.2f} kWh**")
                st.write(f"###### ⏳ Intensité ou Puissance de la période : **{efficience_periode/31:.2f} kW**")
                st.write(f"###### 💰 Coût estimé de la période : **{somme_actuelle * 0.17:.2f} €**")
                st.write(f"######  ℹ️  Durée de la période : {duree} heures/jour")
                
                q1 = data_période_actuel[colonne_p10].quantile(0.25)
                q3 = data_période_actuel[colonne_p10].quantile(0.75)
                iqr = q3 - q1
                seuil_bas = q1 - 1.5 * iqr
                seuil_haut = q3 + 1.5 * iqr
                valeurs_aberrantes = data_période_actuel[
                    (data_période_actuel[colonne_p10] < seuil_bas) |
                    (data_période_actuel[colonne_p10] > seuil_haut)
                ]
                
                color_scale = ['green','mediumseagreen', 'greenyellow','red']
                fig_histogram = px.bar(
                    data_période_actuel,
                    x="Date",
                    y=colonne_p10,
                    title=f'<b>Électricité en {periode}',
                    labels={colonne_p10: f'<b> Énergies (kWh)'},
                    color=colonne_p10,
                    color_continuous_scale=color_scale
                )
                
                fig_histogram.update_layout(
                    font=dict(
                        family="Times New Roman",
                        size=20,
                        color="black"
                    ),
                    title=dict(font=dict(size=20, family="Times New Roman"))
                )
                st.plotly_chart(fig_histogram, use_container_width=True)
                
                key_comment = f"comment_{periode}"
                valeurs_str = "\n".join(
                    [f"- {row['Date'].strftime('%Y-%m-%d %H:%M')} : {row[colonne_p10]:.2f} kWh" for _, row in valeurs_aberrantes.iterrows()]
                )
                texte_par_defaut = f"Données Aberrantes (hors intervalle normal) :\n{valeurs_str}\n\nAnalyse : "
                
                comment = st.text_area(f"📝 Constats", key=key_comment, value=texte_par_defaut, height=150)
                if key_comment not in st.session_state:
                    st.session_state[key_comment] = comment


def graphe_by_analyse():

    # Convertir cal_debut et cal_fin en datetime pour la comparaison
    cal_debut_dt = pd.to_datetime(cal_debut)  # Conversion de la date en datetime
    cal_fin_dt = pd.to_datetime(cal_fin)  # Conversion de la date en datetime
    global colonne_p10
    all_data = analyse()

    # Vérifier si all_data n'est pas None
    if all_data is not None and not all_data.empty:
        # Initialiser la figure pour l'histogramme
        fig_histogram = go.Figure()


        # Récupérer toutes les valeurs uniques de "Période"
        periodes = all_data['Période'].unique()

        # Définir les couleurs pour chaque période
        couleurs_periodes = ["Blue", "MediumSeaGreen", "Red"]
        

        # Boucler sur chaque période pour ajouter des barres de l'histogramme
        for idx, periode in enumerate(periodes):
            data_periode = all_data[(all_data['Période'] == periode) & (all_data[colonne_p10] > 0)]

            # Convertir 'Date' en index de type DatetimeIndex
            data_periode = data_periode.set_index('Date')
            data_periode.index = pd.to_datetime(data_periode.index)

            # Appliquer le resample pour une agrégation par intervalles de 10 minutes
            data_periode_resampled = data_periode[colonne_p10].resample("10min").mean()

            # Ajouter une trace de barre pour chaque période avec sa couleur
            fig_histogram.add_trace(
                go.Bar(
                    x=data_periode_resampled.index,
                    y=data_periode_resampled,
                    name=str(periode),
                    marker=dict(color=couleurs_periodes[idx % len(couleurs_periodes)]),  # Couleur pour chaque période
                    opacity=0.8  # Opacité des barres
                )
            )


        # Personnalisation des titres et des polices pour l'histogramme
        fig_histogram.update_layout(
            title=f'Consommations électriques mesurées : {colonne_p10} sur la période du {cal_debut_dt.date()} au {cal_fin_dt.date()}',
            xaxis_title='Date',
            yaxis_title=f'Consommations en kWh',
            title_font=dict(family="Times New Roman", size=24),
            xaxis_title_font=dict(family="Times New Roman", size=20),
            yaxis_title_font=dict(family="Times New Roman", size=20),
            legend_title_font=dict(family="Times New Roman", size=20),
            legend_font=dict(family="Times New Roman", size=20),
            font=dict(family="Times New Roman", size=20)
        )
        # Afficher l'histogramme dans Streamlit
        st.plotly_chart(fig_histogram)


    else:
        st.error("Aucune donnée à afficher.")

# Fonction pour afficher les graphiques des sommes journalières en courbe, avec les courbes de maximums et minimums sur des pas horaires de 1h
def graphe_ecart_des_mesures():

    global colonne_p10
    all_data = analyse()

    # Vérifier si all_data n'est pas None
    if all_data is not None and not all_data.empty:
        # Initialiser la figure pour les courbes
        fig_line = go.Figure()

        # Récupérer toutes les valeurs uniques de "Période"
        periodes = all_data['Période'].unique()

        # Définir des couleurs spécifiques pour chaque période
        couleurs = ["Yellow", "MediumSeaGreen", "Red", "Blue"]

        # Boucler sur chaque période pour tracer une courbe continue
        for i, periode in enumerate(periodes):
            data_periode = all_data[(all_data['Période'] == periode) & (all_data[colonne_p10] > 0)]

            # Convertir 'Date' en index de type DatetimeIndex
            data_periode = data_periode.set_index('Date')
            data_periode.index = pd.to_datetime(data_periode.index)

            # Appliquer le resample pour une agrégation par intervalles de 1 heure
            data_periode_resampled = data_periode[colonne_p10].resample("30min").agg(['max', 'min'])

            # Couleur pour la période actuelle
            couleur_periode = couleurs[i % len(couleurs)]

            # Ajouter une trace pour la courbe des valeurs maximales de la période (ligne continue)
            fig_line.add_trace(
                go.Scatter(
                    x=data_periode_resampled.index,
                    y=data_periode_resampled['max'],
                    mode='lines',
                    name=f'{periode} - Maximum',
                    line=dict(width=2, color=couleur_periode)  # Ligne continue de couleur de la période
                )
            )

            # Ajouter une trace pour la courbe des valeurs minimales de la période (ligne continue)
            fig_line.add_trace(
                go.Scatter(
                    x=data_periode_resampled.index,
                    y=data_periode_resampled['min'],
                    mode='lines',
                    name=f'{periode} - Minimum',
                    line=dict(width=2, color=couleur_periode)  # Ligne continue de couleur de la période
                )
            )

        # Personnalisation des titres et des polices
        fig_line.update_layout(
            title='Écarts des mesures P10 des consommations électriques',
            xaxis_title='Date',
            yaxis_title=f'Mesures {colonne_p10}',
            title_font=dict(family="Times New Roman", size=24),
            xaxis_title_font=dict(family="Times New Roman", size=20),
            yaxis_title_font=dict(family="Times New Roman", size=20),
            legend_title_font=dict(family="Times New Roman", size=20),
            legend_font=dict(family="Times New Roman", size=18),
            font=dict(family="Times New Roman", size=14),
            colorway=couleurs  # Palette de couleurs pour les périodes
        )

        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig_line)
    else:
        st.error("Aucune donnée à afficher.")

# Fonction principale pour les plages de temps
def analyse():
    global df_energie, colonne_p10
    
    # Filtrer les données pour la période sélectionnée
    debut_periode = cal_debut
    fin_periode = cal_fin

    if df_energie is not None:
        df_energie = df_energie[(df_energie.index.date >= debut_periode) & (df_energie.index.date <= fin_periode)]

        # Définir les périodes fixes
        heures_mode_reduit = (5, 8)
        heures_occupation = (8, 22)

        # Filtrer pour chaque période
        mode_reduit = df_energie[
            (df_energie.index.hour >= heures_mode_reduit[0]) & (df_energie.index.hour < heures_mode_reduit[1])
        ].copy()
        occupation = df_energie[
            (df_energie.index.hour >= heures_occupation[0]) & (df_energie.index.hour < heures_occupation[1])
        ].copy()

        inoccupation_22h_05h = df_energie[
            (df_energie.index.hour >= 22) | (df_energie.index.hour < 5)
        ].copy()

        # Ajouter les étiquettes pour les périodes
        mode_reduit['Période'] = 'Mode Réduit (05h-08h)'
        occupation['Période'] = 'Occupation (08h-22h)'
        inoccupation_22h_05h['Période'] = 'Inoccupation (22h-05h)'

        # Regrouper toutes les périodes
        all_data = pd.concat([mode_reduit, occupation, inoccupation_22h_05h])

        # Ajouter la différence toutes les 10 minutes
        all_data = ajouter_diff_10min(all_data)


        # Retourner les données traitées
        return all_data
    else:
        st.error("Aucune donnée énergétique n'a été trouvée.")
        return None

# Fonction pour afficher les graphiques des consommations par période en nuages de points
def graphe_point_temps():
    global colonne_p10
    all_data = analyse()

    # Vérifier si all_data n'est pas None
    if all_data is not None and not all_data.empty:
        
        # Tracer le graphique en points
        fig_scatter = px.scatter(
            all_data[all_data[colonne_p10] > 0],  # Exclure les valeurs non positives
            x='Date',  # Utiliser la Date sur l'axe des X
            y=colonne_p10,  # Utiliser la consommation comme axe des Y
            color='Période',
            size=colonne_p10,  # Différencier les points par période
            title=f'Évolution des mesures {colonne_p10} pour chaque période',
            labels={f'{colonne_p10}': 'Energie électrique en (kWh)', 'Date': 'Date'},  # Personnalisation des labels
            color_discrete_sequence=["Blue", "MediumSeaGreen", "Red"]  # Discrétisation des couleurs
        )

        # Mise à jour des polices
        fig_scatter.update_layout(
            title_font=dict(family="Times New Roman", size=24),  # Police et taille du titre
            xaxis_title_font=dict(family="Times New Roman", size=20),  # Police et taille de l'axe X
            yaxis_title_font=dict(family="Times New Roman", size=20),  # Police et taille de l'axe Y
            legend_title_font=dict(family="Times New Roman", size=20),  # Police et taille du titre de la légende
            legend_font=dict(family="Times New Roman", size=20),  # Police et taille de la légende
            font=dict(family="Times New Roman", size=14)  # Police générale
        )

        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig_scatter)
    else:
        st.error("Aucune donnée à afficher.")

# Fonction pour afficher les graphiques des consommations en courbe avec les périodes identifiées
def graphe_courbe_all():
    global colonne_p10
    cal_debut_dt = pd.to_datetime(cal_debut)  # Conversion de la date en datetime
    cal_fin_dt = pd.to_datetime(cal_fin)  # Conversion de la date en datetime
    all_data = analyse()
    consommation = all_data[colonne_p10].sum()
    st.write(f"### Consommations de **{consommation:.2f}** kWh mesurées sur la période  du {cal_debut_dt.date()} au {cal_fin_dt.date()}")
            
    # Vérifier si all_data n'est pas None
    if all_data is not None and not all_data.empty:
        # Initialiser la figure pour les courbes
        fig_line = go.Figure()

        # Récupérer toutes les valeurs uniques de "Période"
        periodes = all_data['Période'].unique()

        # Boucler sur chaque période pour tracer une courbe continue
        for periode in periodes:
            data_periode = all_data[(all_data['Période'] == periode) & (all_data[colonne_p10] > 0)]

            # Convertir 'Date' en index de type DatetimeIndex
            data_periode = data_periode.set_index('Date')
            data_periode.index = pd.to_datetime(data_periode.index)

            # Appliquer le resample pour une agrégation par intervalles de 10 minutes
            data_periode_resampled = data_periode[colonne_p10].resample("10min").mean()

            # Ajouter une trace de courbe pour chaque période
            fig_line.add_trace(
                go.Scatter(
                    x=data_periode_resampled.index,  # Utiliser l'index DatetimeIndex après le resample
                    y=data_periode_resampled,
                    mode='lines',
                    name=str(periode),
                    line=dict(width=2)  # Largeur de ligne pour chaque période
                )
            )
        
        # Personnalisation des titres et des polices
        fig_line.update_layout(
            title=f'Evolution du point de mesure : "{colonne_p10}" sur la période du {cal_debut} au {cal_fin}',
            xaxis_title='Date',
            yaxis_title='Consommation (kWh)',
            title_font=dict(family="Times New Roman", size=24),
            xaxis_title_font=dict(family="Times New Roman", size=20),
            yaxis_title_font=dict(family="Times New Roman", size=20),
            legend_title_font=dict(family="Times New Roman", size=20),
            legend_font=dict(family="Times New Roman", size=20),  # Police et taille de la légende
            font=dict(family="Times New Roman", size=14),
            colorway=["Blue", "MediumSeaGreen", "Red"]  # Palette de couleurs
        )

        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig_line)
    else:
        st.error("Aucune donnée à afficher.")

# Fonction pour récupérer les dates de l'interface
def recuperer_dates():
    global cal_debut, cal_fin
    # Entrée date de comparaison pour le choix de l'utilisateur
    cal_debut = st.sidebar.date_input("Entrez la date début à visualiser :", datetime.now().date())
    cal_fin = st.sidebar.date_input("Entrez la date fin à visualiser :", datetime.now().date())

# Fonction pour charger les données des mesures P10
def dieuli_naat_gui(file_path):
    global df_energie, colonne_p10
    colonne_p10 = st.sidebar.text_input("Entrez le nom de la colonne pour les points de 10 minutes (P10) :", "colonne_p10")
    try:
        # Charger la feuille unique
        df_energie = pd.read_excel(file_path)
        col3, col4 = st.columns(2)
        with col3:
            st.write("Information du point de mesure : ", df_energie.columns)
            df_energie['Date'] = pd.to_datetime(df_energie['Date'], errors='coerce')
            df_energie.set_index('Date', inplace=True)
     
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")

# Fonction pour charger les données de température
def dieuli_naat_tangor_gui(file_path):
    global df_temperature, colonne_temperature
    
    # Saisir le nom de la colonne de température
    colonne_temperature = st.sidebar.text_input("Entrez le nom de la colonne pour les températures :", "colonne_temperature")
    
    try:
        # Charger le fichier Excel
        df_temperature = pd.read_excel(file_path)
        st.write("Information du point de mesure : ", df_temperature.columns)
        # Convertir la colonne 'Date' en type datetime
        df_temperature['Date'] = pd.to_datetime(df_temperature['Date'], errors='coerce')
        df_temperature.set_index('Date', inplace=True)
        
        # Vérifier si la colonne de température existe
        if colonne_temperature in df_temperature.columns:
            # Nettoyer les valeurs en supprimant les espaces, guillemets et en remplaçant les virgules par des points
            df_temperature[colonne_temperature] = (
                df_temperature[colonne_temperature]
                .astype(str)
                .str.replace(",", ".", regex=False)  # Remplacer les virgules pour les décimales
                .str.extract(r"(-?\d+\.?\d*)")[0]  # Extraire les valeurs numériques correctes
            )

            # Convertir les valeurs en float, avec des erreurs ignorées (les valeurs incorrectes deviendront NaN)
            df_temperature[colonne_temperature] = pd.to_numeric(df_temperature[colonne_temperature], errors='coerce')
            
            # Remplacer les valeurs NaN par la médiane de la colonne
            median_temperature = df_temperature[colonne_temperature].median()
            df_temperature[colonne_temperature].fillna(median_temperature, inplace=True)
        else:
            st.warning("La colonne spécifiée pour les températures n'existe pas dans le fichier.")
            return
            
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")

# Fonction principale pour le calcul de talon
def talon():
    global df_energie
    df_energie['Heure'] = df_energie.index.hour

    Talon_Nocturne = df_energie[(df_energie["Heure"] >= 22) | (df_energie["Heure"] < 5)].copy()
    Talon_mode_reduit = df_energie[(df_energie["Heure"] >= 5) & (df_energie["Heure"] < 8)].copy()
    Talon_diurne = df_energie[(df_energie["Heure"] >= 8) & (df_energie["Heure"] < 22)].copy()

    sommes = [calculer_sommes(talon) for talon in [Talon_Nocturne, Talon_mode_reduit, Talon_diurne]]
    somme_jour = pd.concat([s[0] for s in sommes], axis=1)
    somme_jour.columns = ['Talon Nocturne (22h - 05h)', 'Talon en mode Réduit (05h - 08h)', 'Talon Diurne (08h - 22h)']

    # Concatenation des talons
    les_talons_combined = pd.concat([Talon_Nocturne[colonne_p10], Talon_mode_reduit[colonne_p10], Talon_diurne[colonne_p10]], axis=1)
    
    # Renommer les colonnes pour une meilleure lisibilité
    les_talons_combined.columns = ['Talon Nocturne (22h - 05h)', 'Talon en mode Réduit (05h - 08h)', 'Talon Diurne (08h - 22h)']
    
    # Conserver également la colonne Date dans les_talons_combined
    les_talons_combined['Date'] = les_talons_combined.index.date  # Conserver la date si nécessaire

    return somme_jour, les_talons_combined


# Fonction pour afficher les graphiques 
def afficher_graphiques():
    # Obtenir les sommes des périodes depuis la fonction talon
    somme_jour, les_talons_combined = talon()

    # Assurez-vous que cal_debut et cal_fin sont valides
    if cal_debut is None or cal_fin is None:
        st.error("Veuillez sélectionner une plage de dates valide.")
        return  # Sortir si les dates ne sont pas valides

    # Convertir cal_debut et cal_fin en datetime pour la comparaison
    cal_debut_dt = pd.to_datetime(cal_debut)  # Conversion de la date en datetime
    cal_fin_dt = pd.to_datetime(cal_fin)  # Conversion de la date en datetime
    
    # Filtrer les données selon les dates dans les_talons_combined
    les_talons_combined_selection = les_talons_combined[
        (les_talons_combined.index >= cal_debut_dt) & (les_talons_combined.index <= cal_fin_dt)
    ]

    if les_talons_combined_selection.empty:
        st.error("Aucune donnée disponible pour la période sélectionnée.")
        return

    # Ajout d'une colonne 'Date' pour le tracé
    les_talons_combined_selection['Date'] = les_talons_combined_selection.index  # Conserver la date
    # Définir les colonnes à tracer
    colonnes_a_tracer = [
        'Talon en mode Réduit (05h - 08h)',
        'Talon Diurne (08h - 22h)',
        'Talon Nocturne (22h - 05h)'
    ]

    # Utilisation de la méthode melt pour préparer les données pour le tracé
    melted_data_talons = les_talons_combined_selection.melt(id_vars=['Date'], value_vars=colonnes_a_tracer, var_name='Période', value_name='Valeur')
    
    # Vérification que chaque période est représentée
    if melted_data_talons.empty:
        st.error("Aucune donnée disponible après réorganisation.")
        return

    # Filtrer les données pour la période sélectionnée
    periode_selection = les_talons_combined_selection[(les_talons_combined_selection.index >= cal_debut_dt) & (les_talons_combined_selection.index <= cal_fin_dt)]

    # Ajouter une colonne 'Date' pour chaque période
    periode_selection['Date'] = periode_selection.index

    # Réorganiser les données pour le tracé avec un DataFrame 'melted'
    les_talons_combined_selection = periode_selection.melt(id_vars=['Date'], value_vars=colonnes_a_tracer, var_name='Période', value_name='Valeur')

    # Convertir 'Valeur' en numérique et gérer les NaN
    les_talons_combined_selection['Valeur'] = pd.to_numeric(les_talons_combined_selection['Valeur'], errors='coerce')
    les_talons_combined_selection = les_talons_combined_selection.dropna(subset=['Valeur'])  # Supprimer les lignes avec NaN

    # Remplir les dates manquantes pour chaque période afin d'assurer une continuité
    # En pivotant puis remplissant les dates, toutes les périodes auront une ligne pour chaque date, même si la valeur est NaN
    les_talons_combined_selection_pivot = melted_data_talons.pivot_table(
        index='Date', 
        columns='Période', 
        values='Valeur'
    ).reset_index()

    # Réarranger le DataFrame pour le format 'long' à nouveau après remplissage des périodes manquantes
    les_talons_combined_selection_filled = les_talons_combined_selection_pivot.melt(
        id_vars=['Date'], 
        var_name='Période', 
        value_name='Valeur'
    )

    # Assurer la suppression des NaN dans la version finale avant le tracé (vous pouvez choisir de les interpoler si vous préférez)
    les_talons_combined_selection_filled_clean = les_talons_combined_selection_filled.fillna(0)  # Remplacer les NaN par 0 pour maintenir la continuité
    
    return les_talons_combined_selection_filled_clean, les_talons_combined_selection, melted_data_talons, colonnes_a_tracer

# Fonction pour afficher les graphiques des sommes journalières en Bar
def graphe_bar():
    global colonne_p10
    cal_debut_dt = pd.to_datetime(cal_debut)  # Conversion de la date en datetime
    cal_fin_dt = pd.to_datetime(cal_fin)  # Conversion de la date en datetime
    result = afficher_tableaux_et_graphiques_par_periode()
    melted_data = result['melted_data']
    
    # Calculer la somme pour chaque période et année
    sums_per_period = (
        melted_data[melted_data['Valeur'] > 0]
        .groupby(['Période', 'Année'])['Valeur']
        .sum()
        .reset_index()
    )
    
    # Tracer le graphique avec Plotly
    fig_bar = px.bar(
        melted_data[melted_data['Valeur'] > 0],  # Exclure les valeurs non positives
        x='Période',  
        y='Valeur',  
        color='Année',  
        barmode='group',  
        title=f'Comparaison des consommations électriques du {cal_debut_dt.date()} au {cal_fin_dt.date()} par la période précédente : {colonne_p10}',
        labels={'Valeur': 'Mesures Electriques en (kWh)', 'Période': 'Consommation triée par période de fonctionnement'},
        color_discrete_sequence=['SeaGreen', 'LightGreen']  # Utilisation des couleurs Teal et LightSeaGreen
    )

    # Associer les couleurs des périodes aux annotations
    colors = {trace['name']: trace['marker']['color'] for trace in fig_bar['data']}
    
    # Ajouter des annotations pour chaque somme avec un offset fixe entre chaque période
    last_period = None
    yshift = 0
    for idx, row in sums_per_period.iterrows():
        # Vérifier si la période est différente de la précédente
        if row['Période'] != last_period:
            yshift = 0  # Réinitialiser le décalage pour une nouvelle période
            last_period = row['Période']
        
        # Trouver la couleur associée à l'année
        color = colors.get(str(row['Année']), 'black')  # Noir par défaut si non trouvé
        
        # Ajouter une annotation pour chaque barre
        fig_bar.add_annotation(
            x=row['Période'],  # Position X : alignement avec la période
            y=row['Valeur'],  # Position Y : somme
            text=f"{row['Valeur']:.2f} kWh",  # Texte : somme en kWh
            showarrow=False,  # Pas de flèche
            font=dict(size=18, color=color),  # Style de texte avec couleur
            align='center',  # Alignement horizontal centré
            yshift= 45 + yshift  # Décalage vertical initial de 10 + 15 par annotation
        )
        
        # Augmenter le décalage pour la prochaine annotation de la même période
        yshift += 15

    # Mise à jour des polices et rendre le fond transparent
    fig_bar.update_layout(
        title_font=dict(family="Times New Roman", size=24),  # Taille et police du titre
        xaxis_title_font=dict(family="Times New Roman", size=24),  # Taille et police du titre de l'axe X
        yaxis_title_font=dict(family="Times New Roman", size=24),  # Taille et police du titre de l'axe Y
        legend_title_font=dict(family="Times New Roman", size=24),  # Taille et police du titre de la légende
        legend_font=dict(family="Times New Roman", size=20),  # Police et taille de la légende
        font=dict(family="Times New Roman", size=24),  # Taille et police générale du texte
        plot_bgcolor='rgba(0,0,0,0)',  # Fond du graphique transparent
        paper_bgcolor='rgba(0,0,0,0)'  # Fond du papier (hors graphique) transparent
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig_bar)

# Fonction pour afficher les graphiques des sommes journalières en Courbe 
def graphe_courbe():
    global colonne_p10
    cal_debut_dt = pd.to_datetime(cal_debut)  # Conversion de la date en datetime
    cal_fin_dt = pd.to_datetime(cal_fin)  # Conversion de la date en datetime
    result = afficher_tableaux_et_graphiques_par_periode()
    melted_data1 = result['melted_data1']
    
    # Tracer les courbes avec Date sur l'axe des X
    fig_line = px.line(
        melted_data1[melted_data1['Valeur'] > 0],  # Exclure les valeurs non positives
        x='Date',  # Utiliser la Date sur l'axe des X
        y='Valeur',  # Utiliser la consommation comme axe des Y
        color='Période',  # Différencier les courbes par période
        title=f"Évolution des consommations par période du {cal_debut_dt.date()} au {cal_fin_dt.date()} : {colonne_p10}",
        labels={'Valeur': 'Consommation en kWh', 'Date': 'Date'},  # Personnalisation des labels
        color_discrete_sequence=["Red", "Blue", "MediumSeaGreen"]  # Discrétisation des couleurs
    )

    # Mise à jour des polices
    fig_line.update_layout(
        title_font=dict(family="Times New Roman", size=24),  # Police et taille du titre
        xaxis_title_font=dict(family="Times New Roman", size=20),  # Police et taille de l'axe X
        yaxis_title_font=dict(family="Times New Roman", size=20),  # Police et taille de l'axe Y
        legend_title_font=dict(family="Times New Roman", size=20),  # Police et taille du titre de la légende
        legend_font=dict(family="Times New Roman", size=20),  # Police et taille de la légende
        font=dict(family="Times New Roman", size=14)  # Police générale
        
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig_line)

# Fonction pour dérouler l'évolution du talon élctrique en Replay afin d'identifier les dérives
# Permettant l'utilisateur de définir ces propres seuils pour chaque période du talon électrique 
def graphe_interactif():
    cal_debut_dt = pd.to_datetime(cal_debut)  # Conversion de la date en datetime
    cal_fin_dt = pd.to_datetime(cal_fin)  # Conversion de la date en datetime
    # Simule l'obtention des données avec les colonnes 'Date', 'Valeur', 'Période'
    les_talons_combined_selection_filled_clean, les_talons_combined_selection, melted_data_talons, colonnes_a_tracer = afficher_graphiques() 

    # Saisie des seuils pour chaque période
    seuils = {} # dictionnaire des seuils
    periodes = les_talons_combined_selection_filled_clean['Période'].unique()
    
    st.sidebar.write("Définissez les seuils pour chaque période:")
    for periode in periodes: # boucle parcourant les périodes pour créer les entrées des seuils
        seuils[periode] = float(st.sidebar.text_input(f"Seuil pour la période {periode} :", "10"))

    # Ajouter une colonne "Couleur" pour respecter les couleurs des périodes dans la légende
    les_talons_combined_selection_filled_clean['Couleur'] = les_talons_combined_selection_filled_clean['Période']

    # Création du graphique avec les couleurs correspondant aux périodes
    fig_interactif = px.scatter(
        les_talons_combined_selection_filled_clean, 
        x="Date",  # Axe X pour illustrer l'évolution temporelle
        y="Valeur",  # Axe Y pour les valeurs de consommation
        color="Période",  # Couleur basée sur les périodes pour la légende
        hover_name="Période",  # Nom de la période au survol
        animation_frame="Date",  # Animation basée sur la date
        animation_group="Période",  # Groupe d'animation par période
        size='Valeur',  # Taille des points proportionnelle à la valeur
        range_y=[les_talons_combined_selection_filled_clean['Valeur'].min() * 0.9, 
                 les_talons_combined_selection_filled_clean['Valeur'].max() * 1.1],  # Plage Y
        range_x=[les_talons_combined_selection_filled_clean['Date'].min(), 
                 les_talons_combined_selection_filled_clean['Date'].max()],  # Plage X
        title=f'Évolution animée des consommations par période du {cal_debut_dt.date()} au {cal_fin_dt.date()}',
        color_discrete_sequence=["MediumSeaGreen", "Red", "Blue"]  # Couleurs pour les périodes
    )

    # Ajout de lignes entre les points
    fig_interactif.update_traces(mode='lines+markers')

    # Mise à jour des axes et des polices
    fig_interactif.update_layout(
        title_font=dict(family="Times New Roman", size=20),
        xaxis_title_font=dict(family="Times New Roman", size=18),
        yaxis_title_font=dict(family="Times New Roman", size=18),
        legend_title_font=dict(family="Times New Roman", size=18),
        legend_font=dict(family="Times New Roman", size=18),  # Police et taille de la légende
        font=dict(family="Times New Roman", size=16)
    )

    # Ajouter une trace pour les points qui dépassent le seuil et les faire clignoter en rouge
    for i, row in les_talons_combined_selection_filled_clean.iterrows():
        if row['Valeur'] >= seuils[row['Période']]:
            # Ajout de points supplémentaires en rouge sans affecter la légende des périodes
            fig_interactif.add_trace(go.Scatter(
                x=[row['Date']], 
                y=[row['Valeur']],
                mode='markers',
                marker=dict(color='Magenta', size=15, symbol='circle', opacity=0.3),
                name=f"Dérive: {row['Période']} (>{seuils[row['Période']]} kWh {row['Date']})",
                showlegend=True  # afficher cette trace dans la légende
            ))

    # Afficher le graphique interactif dans Streamlit
    st.plotly_chart(fig_interactif)

# Fonction pour afficher les graphiques des consommations en Tunnel avec les périodes identifiées 
def graphe_tunnel():
    cal_debut_dt = pd.to_datetime(cal_debut)  # Conversion de la date en datetime
    cal_fin_dt = pd.to_datetime(cal_fin)  # Conversion de la date en datetime
    les_talons_combined_selection_filled_clean, les_talons_combined_selection, melted_data_talons, colonnes_a_tracer=afficher_graphiques()
    # Création du graphique en entonnoir (funnel chart) avec Plotly Express
    fig_tunnel = px.funnel(
        les_talons_combined_selection_filled_clean, 
        x='Valeur', 
        y='Période', 
        color="Période",
        color_discrete_sequence=["MediumSeaGreen", "Red", "Blue"],  # Discrétisation des couleurs
        hover_data= ["Valeur", "Date"]
        
    )

    # Mise à jour de la mise en page du graphique
    fig_tunnel.update_layout(
        xaxis_title="Consommation d'énergie (Valeur)",  # Ajout du titre de l'axe X
        yaxis_title="Période de la journée",  # Ajout du titre de l'axe Y
        title_font=dict(family="Times New Roman", size=28),  # Taille et police du titre du graphique
        xaxis_title_font=dict(family="Times New Roman", size=24),  # Taille et police du titre de l'axe X
        yaxis_title_font=dict(family="Times New Roman", size=24),  # Taille et police du titre de l'axe Y
        legend_title_font=dict(family="Times New Roman", size=24),  # Taille et police du titre de la légende
        legend_font=dict(family="Times New Roman", size=20),  # Police et taille de la légende
        font=dict(family="Times New Roman", size=18),  # Police et taille générale du texte
        funnelmode='stack',  # Empilage des sections du funnel
        title_text=f"Consommation d'énergie par Période du {cal_debut_dt.date()} au {cal_fin_dt.date()}",  # Titre du graphique
    )

    # Mise à jour des axes pour ajuster les tailles de texte des ticks et des labels
    fig_tunnel.update_xaxes(tickfont=dict(family='Times New Roman', size=24))
    fig_tunnel.update_yaxes(tickfont=dict(family='Times New Roman', size=24))

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig_tunnel)

# Fonction pour afficher les graphiques des consommations en violon avec les périodes identifiées 
def graphe_violon():
    les_talons_combined_selection_filled_clean, les_talons_combined_selection, melted_data_talons, colonnes_a_tracer=afficher_graphiques()  
    # Convertir cal_debut et cal_fin en datetime pour la comparaison
    cal_debut_dt = pd.to_datetime(cal_debut)  # Conversion de la date en datetime
    cal_fin_dt = pd.to_datetime(cal_fin)  # Conversion de la date en datetime
    # Tracer le graphique en violon
    fig_violon = px.violin(
        melted_data_talons[melted_data_talons['Valeur'] > 0],  # Exclure les valeurs non positives
        y='Valeur', 
        x='Période', 
        color='Période', 
        box=True,
        points='all',  # Afficher tous les points
        title="Distribution de la consommation énergétique (kWh) entre {} et {}".format(cal_debut_dt.date(), cal_fin_dt.date()),
        color_discrete_sequence=["Blue", "MediumSeaGreen", "Red"]  # Discrétisation des couleurs 
    )
    # Mise à jour des polices
    fig_violon.update_layout(
        title_font=dict(family="Times New Roman", size=20),  # Police et taille du titre
        xaxis_title_font=dict(family="Times New Roman", size=20),  # Police et taille de l'axe X
        yaxis_title_font=dict(family="Times New Roman", size=20),  # Police et taille de l'axe Y
        legend_title_font=dict(family="Times New Roman", size=20),  # Police et taille du titre de la légende
        legend_font=dict(family="Times New Roman", size=20),  # Police et taille de la légende
        font=dict(family="Times New Roman", size=14)  # Police générale
        
    )
    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig_violon)

# Fonction pour afficher les graphiques des consommations en circulaire avec les périodes identifiées  
def graphe_pie():
    les_talons_combined_selection_filled_clean, les_talons_combined_selection, melted_data_talons, colonnes_a_tracer=afficher_graphiques() 
    # Création du graphique en camembert (pie chart) avec Plotly Express
    fig_pie = px.pie(
        les_talons_combined_selection_filled_clean, 
        values='Valeur', 
        names='Période', 
        title=f'Répartition des consommations électriques mesurées: {colonne_p10}',
        color='Période',  # Correction de la position de color
        color_discrete_sequence=["MediumSeaGreen", "Red", "Blue"]  # Discrétisation des couleurs
    )

    # Mise à jour de la mise en page du graphique
    fig_pie.update_layout(
        title_font=dict(family="Times New Roman", size=20),  # Taille et police du titre du graphique
        legend_title_font=dict(family="Times New Roman", size=20),  # Taille et police du titre de la légende
        font=dict(family="Times New Roman", size=14),  # Police et taille générale du texte
        legend=dict(
            font=dict(size=20),  # Taille de police des éléments de la légende
            orientation="h",  # Affichage horizontal de la légende
            yanchor="bottom",
            y=-0.3,  # Positionnement de la légende en dessous du graphique
            xanchor="center",
            x=0.5
        )
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig_pie)

# Condition des évènements
if mode == "Analyse et rapport":
    if uploaded_file or uploaded_file_temperature:
        dieuli_naat_tangor_gui(uploaded_file_temperature)
        dieuli_naat_gui(uploaded_file)
        recuperer_dates()
        by_analyse()
        graphe_by_analyse() 
        thiambar()
        graphe_pie()
        graphe_violon()
        graphe_point_temps()
        graphe_regression_tempo()
        graphe_regression_lineaire()
    
        # Créer deux colonnes pour les afficher côte à côte
        col3, col4 = st.columns(2)
    else:
        st.error("Veuillez charger un fichier Excel.")        
elif mode == "Export des données":
    # Fonction pour les Consommations annuelles avec annotations des évolutions de 2022 à 2024
    def tracer_barres_annuelles_avec_annotations(df, villes):
        df_annuel = df.groupby(['Année'])[villes].sum().reset_index()
        st.write("### Diagrammes en barres : Consommations annuelles avec évolutions (2022-2024)")

        for ville in villes:
            # Calcul de l'évolution d'une année à l'autre
            df_annuel[f'Évolution_{ville}'] = df_annuel[ville].pct_change() * 100

            # Tracer le diagramme en barres
            fig = px.bar(
                df_annuel,
                x='Année',
                y=ville,
                title=f"Evolution des consommations électriques annuelles par rapport à l'année précédente : {ville}",
                text=df_annuel[ville].apply(lambda x: f"{x:.0f} kWh"),  # Afficher les consommations avec 2 chiffres après la virgule
                labels={ville: 'Consommation', 'Année': 'Année'},
                color_discrete_sequence=["MediumAquamarine"]
            )

            # Ajouter des annotations pour l'évolution
            for i, row in df_annuel.iterrows():
                if i > 0:  # Pas d'évolution pour la première année
                    # Déterminer la couleur de l'annotation en fonction de l'évolution
                    couleur_evolution = 'red' if row[f'Évolution_{ville}'] > 0 else 'seagreen'

                    # Ajouter l'annotation avec flèche pour l'évolution
                    fig.add_annotation(
                        x=row['Année'],
                        y=row[ville],
                        text=f"{row[f'Évolution_{ville}']:.0f}%",  # Afficher l'évolution avec 2 chiffres après la virgule
                        showarrow=True,  # Flèche activée
                        arrowhead=2,  # Type de flèche
                        ax=0,  # Déplacement de l'annotation horizontalement
                        ay=-30,  # Déplacement de l'annotation verticalement
                        font=dict(size=24, color=couleur_evolution)  # Couleur conditionnelle
                    )

            # Mettre à jour la mise en page pour formater les années sans virgules
            fig.update_layout(
                xaxis=dict(
                    tickmode='linear',  # Utiliser un mode linéaire pour l'axe des X
                    tick0=df_annuel['Année'].min(),  # Début de l'axe des X
                    dtick=1,  # Espacement des ticks (1 an)
                    ticks="outside",  # Ajouter des ticks à l'extérieur
                    tickvals=df_annuel['Année'],  # Spécifier les années à afficher
                    ticktext=[str(int(val)) for val in df_annuel['Année']]  # Convertir les années en entiers
                )
            )

            st.plotly_chart(fig)
    # Fonction 2 : Consommations mensuelles de 2024 avec annotations des évolutions par rapport à 2023
    def tracer_barres_mensuelles_2024_avec_annotations(df, villes):
        # Fusionner les données de 2023 et 2024 pour chaque mois
        df_2023 = df[df['Année'] == 2023].groupby('Mois')[villes].sum().reset_index()
        df_2024 = df[df['Année'] == 2024].groupby('Mois')[villes].sum().reset_index()

        st.write("### Diagrammes en barres : Consommations mensuelles de 2024 avec évolutions par rapport à 2023")
        
        for ville in villes:
            # Fusionner les consommations de 2023 et 2024
            df_comparatif = df_2024.copy()
            df_comparatif['Consommation_2023'] = df_2023[ville]
            df_comparatif['Évolution'] = ((df_comparatif[ville] - df_comparatif['Consommation_2023']) / 
                                        df_comparatif['Consommation_2023']) * 100

            # Tracer le diagramme en barres comparatif entre 2023 et 2024
            fig = px.bar(
                df_comparatif,
                x='Mois',
                y=[ville, 'Consommation_2023'],  # Ajouter les deux années côte à côte
                title=f"Consommations mensuelles de {ville} en 2024 vs 2023",
                text_auto=True,  # Afficher automatiquement les consommations sur les barres
                labels={'Mois': 'Mois', 'value': 'Consommation', 'variable': 'Année'},
                color='variable',  # Différencier les années par couleur
                color_discrete_sequence=["MediumAquamarine", "Teal"]  # Couleurs distinctes
            )

            # Ajouter des annotations pour les évolutions
            for i, row in df_comparatif.iterrows():
                couleur_evolution = 'red' if row['Évolution'] > 0 else 'seagreen'

                fig.add_annotation(
                    x=row['Mois'],
                    y=max(row[ville], row['Consommation_2023']) + (0.05 * max(row[ville], row['Consommation_2023'])),
                    text=f"{row['Évolution']:.0f}%",
                    showarrow=True,
                    font=dict(size=14, color=couleur_evolution)  # Appliquer la couleur conditionnelle
                )

            # Formatage de l'axe des X pour afficher les noms des mois
            mois_noms = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=df_comparatif['Mois'],  # Utiliser les indices des mois
                    ticktext=mois_noms  # Remplacer les numéros par les noms des mois
                ),
                barmode='group',  # Afficher les barres côte à côte
                yaxis_showticklabels=False,  # Masquer les labels de l'axe des Y
                yaxis_visible=False  # Masquer l'axe des Y
            )

            st.plotly_chart(fig)

    def tracer_barres_trimestrielles_avec_annotations(df, villes):
            # Créer une colonne 'Trimestre' basée sur le mois
            df['Trimestre'] = ((df['Mois'] - 1) // 3) + 1

            # Grouper par trimestre et année
            df_trimestriel = df.groupby(['Année', 'Trimestre'])[villes].sum().reset_index()

            # Créer une nouvelle colonne 'Période' combinant Trimestre et Année
            df_trimestriel['Période'] = 'T' + df_trimestriel['Trimestre'].astype(str) + ' ' + df_trimestriel['Année'].astype(str)

            st.write("### Diagrammes en barres : Consommations trimestrielles avec évolutions (2022-2024)")

            # Créer un dictionnaire pour mapper les couleurs (chaque année avec une couleur spécifique)
            color_map = {
                2022: "MediumAquamarine",
                2023: "green",
                2024: "lightgreen"
            }

            for ville in villes:
                # Calculer l'évolution par rapport à l'année précédente (2022 vs 2021, 2023 vs 2022, 2024 vs 2023)
                df_trimestriel[f'Évolution_{ville}'] = df_trimestriel.groupby('Trimestre')[ville].pct_change() * 100

                # Tracer le diagramme en barres
                fig = px.bar(
                    df_trimestriel,
                    x='Période',
                    y=ville,
                    color='Année',
                    title=f"Consommation trimestrielle de {ville} (2022-2024)",
                    text=df_trimestriel[ville].apply(lambda x: f"{x:.0f} kWh"),  # Ne pas afficher de consommation sur les barres
                    labels={ville: 'Consommation électrique (kWh)', 'Période': 'Période'},
                    color_discrete_map=color_map  # Appliquer les couleurs définies dans color_map
                )

                # Ajouter des annotations pour l'évolution sur les barres
                for i, row in df_trimestriel.iterrows():
                    # Ne pas afficher d'annotations pour l'année 2021
                    if row['Année'] != 2021:
                        # Ajouter l'annotation pour l'évolution avec une couleur en fonction de l'augmentation/diminution
                        couleur_evolution = 'red' if row[f'Évolution_{ville}'] > 0 else 'seagreen'
                        fig.add_annotation(
                            x=row['Période'],
                            y=row[ville],
                            text=f"{row[f'Évolution_{ville}']:.0f}%",  # Afficher l'évolution avec 2 chiffres après la virgule
                            showarrow=True,
                            arrowhead=2,
                            ax=0,
                            ay=-30,
                            font=dict(size=12, color=couleur_evolution)
                        )

                # Masquer l'échelle des Y
                fig.update_layout(
                    yaxis_showticklabels=False,  # Masquer les labels de l'axe des Y
                    yaxis_visible=False  # Masquer l'axe des Y
                )

                # Afficher le graphique dans Streamlit
                st.plotly_chart(fig)   
    
    def tracer_jauges_annuelles(df, villes):
            """
            Trace des jauges pour représenter l'évolution annuelle (2024 par rapport à 2023) 
            pour chaque ville dans le DataFrame.
            """
            st.write("### Jauges : Évolution annuelle (2024 vs 2023)")

            # Filtrer les données pour les années 2023 et 2024
            df_filtre = df[df['Année'].isin([2023, 2024])]

            # Groupement par année et calcul des sommes par ville
            df_annuel = df_filtre.groupby(['Année'])[villes].sum().reset_index()

            for ville in villes:
                # Calculer l'évolution en pourcentage de 2024 par rapport à 2023
                consommation_2023 = df_annuel.loc[df_annuel['Année'] == 2023, ville].sum()
                consommation_2024 = df_annuel.loc[df_annuel['Année'] == 2024, ville].sum()

                if consommation_2023 == 0:
                    evolution = 0
                else:
                    evolution = ((consommation_2024 - consommation_2023) / consommation_2023) * 100

                # Déterminer la couleur de la flèche en fonction de l'évolution
                couleur_fleche = 'green' if evolution < 0 else 'red'

                # Limiter l'évolution affichée à un intervalle [-30, 30] pour une meilleure visualisation
                evolution_limitee = max(min(evolution, 30), -30)

                # Créer la jauge avec Plotly
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=evolution_limitee,
                    title={'text': f""},
                    delta={'reference': 0, 'position': "top", 'increasing': {'color': 'red'}, 'decreasing': {'color': 'green'}},
                    gauge={
                        'axis': {'range': [-30, 30], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': couleur_fleche},
                        'steps': [
                            {'range': [-30, 0], 'color': "lightgreen"},
                            {'range': [0, 30], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': couleur_fleche, 'width': 4},
                            'thickness': 0.75,
                            'value': evolution_limitee
                        }
                    },
                    # Ajouter le % directement après la valeur de la jauge
                    number={'suffix': '%', 'font': {'size': 20, 'color': 'black'}}
                ))

                # Mise en page
                fig.update_layout(
                    height=200,
                    width=400,
                    margin=dict(l=50, r=50, t=50, b=50),
                    title={
                        'text': f"Évolution Electricité  (2024 vs 2023) : {ville}",
                        'x': 0.5,
                        'xanchor': 'center'
                    }
                )

                # Afficher la jauge dans Streamlit
                st.plotly_chart(fig)


    # Charger le fichier Excel téléchargé
    uploaded_file = st.file_uploader("Téléchargez un fichier Excel", type=["xlsx"])

    if uploaded_file is not None:
        # Lire le fichier Excel dans un DataFrame
        df = pd.read_excel(uploaded_file)

        # Afficher les données dans une table
        st.write("### Aperçu des données :")
        st.dataframe(df)
        st.write(df.columns[:])  # Affiche les colonnes récupérées


        # Convertir la colonne 'Mois-Année' en type datetime pour un meilleur tri (si nécessaire)
        try:
            df['Mois-Année'] = pd.to_datetime(df['Mois-Année'], format='%b-%y')
            df = df.sort_values(by='Mois-Année')
        except Exception as e:
            st.warning(f"Impossible de convertir la colonne 'Mois-Année' en format date : {e}")

        # Ajouter les colonnes Année et Mois
        df['Année'] = df['Mois-Année'].dt.year
        df['Mois'] = df['Mois-Année'].dt.month

        # Consommations par ville
        villes = df.columns[1:-2]  # Exclure les colonnes non pertinentes (comme 'Mois-Année', 'Année', etc.)
        tracer_barres_annuelles_avec_annotations(df, villes)
        tracer_barres_mensuelles_2024_avec_annotations(df, villes)
        tracer_barres_trimestrielles_avec_annotations(df, villes)
        tracer_jauges_annuelles(df, villes)


