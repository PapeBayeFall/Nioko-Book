import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# Configurer la mise en page pour utiliser toute la largeur disponible
st.set_page_config(layout="wide")
# Configuration du thème seaborn
sns.set_theme(style="whitegrid")
# Charger le fichier Excel téléchargé
uploaded_file = st.file_uploader("Téléchargez un fichier Excel", type=["xlsx"])

if uploaded_file is not None:
    # Lire le fichier Excel dans un DataFrame
    df = pd.read_excel(uploaded_file)

    # Afficher les données dans une table
    st.write("### Aperçu des données :")
    st.dataframe(df)

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

    # Fonction 1 : Consommations annuelles avec annotations des évolutions de 2022 à 2024
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
                title=f"Consommation annuelle de {ville} (2023-2025)",
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
                        font=dict(size=18, color=couleur_evolution)  # Couleur conditionnelle
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
        df_2023 = df[df['Année'] == 2024].groupby('Mois')[villes].sum().reset_index()
        df_2024 = df[df['Année'] == 2025].groupby('Mois')[villes].sum().reset_index()

        st.write("### Diagrammes en barres : Consommations mensuelles de 2024 avec évolutions par rapport à 2023")
        
        for ville in villes:
            # Fusionner les consommations de 2024 et 2025
            df_comparatif = df_2024.copy()
            df_comparatif['Consommation_2024'] = df_2023[ville]
            df_comparatif['Évolution'] = ((df_comparatif[ville] - df_comparatif['Consommation_2024']) / 
                                        df_comparatif['Consommation_2024']) * 100

            # Tracer le diagramme en barres comparatif entre 2024 et 2024
            fig = px.bar(
                df_comparatif,
                x='Mois',
                y=[ville, 'Consommation_2024'],  # Ajouter les deux années côte à côte
                title=f"Consommations mensuelles de {ville} en 2025 vs 2024",
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
                    y=max(row[ville], row['Consommation_2024']) + (0.05 * max(row[ville], row['Consommation_2024'])),
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
            2023: "MediumAquamarine",
            2024: "green",
            2025: "lightgreen"
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
                title=f"{ville} : Evolution des Consommations trimestrielles (2023-2025)",
                text=df_trimestriel[ville].apply(lambda x: f"{x:.0f} kWh"),  # Ne pas afficher de consommation sur les barres
                labels={ville: 'Consommation électrique (kWh)', 'Période': 'Période'},
                color_discrete_map=color_map  # Appliquer les couleurs définies dans color_map
            )

            # Ajouter des annotations pour l'évolution sur les barres
            for i, row in df_trimestriel.iterrows():
                # Ne pas afficher d'annotations pour l'année 2021
                if row['Année'] != 2022:
                    # Ajouter l'annotation pour l'évolution avec une couleur en fonction de l'augmentation/diminution
                    couleur_evolution = 'red' if row[f'Évolution_{ville}'] > 0 else 'seagreen'
                    fig.add_annotation(
                        x=row['Période'],
                        y=row[ville],
                        text=f"{row[f'Évolution_{ville}']:.2f}%",  # Afficher l'évolution avec 2 chiffres après la virgule
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

    def tracer_barres_trimestrielles(df, villes):
        # Créer une colonne 'Trimestre' basée sur le mois
        df['Trimestre'] = ((df['Mois'] - 1) // 3) + 1

        # Grouper par trimestre et année
        df_trimestriel = df.groupby(['Année', 'Trimestre'])[villes].sum().reset_index()

        # Créer une nouvelle colonne 'Période' combinant Trimestre et Année
        df_trimestriel['Période'] = 'T' + df_trimestriel['Trimestre'].astype(str) + ' ' + df_trimestriel['Année'].astype(str)

        st.write("### Diagrammes en barres : Evolution des Consommations trimestrielles (2023-2025)")

        # Créer un dictionnaire pour mapper les couleurs (chaque année avec une couleur spécifique)
        color_map = {
            2023: "MediumAquamarine",
            2024: "green",
            2025: "lightgreen"
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
                title=f"{ville} : Evolution des Consommations trimestrielles (2023-2025)",
                text=df_trimestriel[ville].apply(lambda x: f"{x:.2f} kWh"),  # Ne pas afficher de consommation sur les barres
                labels={ville: 'Consommation électrique (kWh)', 'Période': 'Période'},
                color_discrete_map=color_map  # Appliquer les couleurs définies dans color_map
            )

            # Ajouter des annotations pour l'évolution sur les barres
            for i, row in df_trimestriel.iterrows():
                # Ne pas afficher d'annotations pour l'année 2021
                if row['Année'] != 2022:
                    # Ajouter l'annotation pour l'évolution avec une couleur en fonction de l'augmentation/diminution
                    couleur_evolution = 'red' if row[f'Évolution_{ville}'] > 0 else 'seagreen'
                    fig.add_annotation(
                        x=row['Période'],
                        y=row[ville],
                        text=f"{row[f'Évolution_{ville}']:.2f}%",  # Afficher l'évolution avec 2 chiffres après la virgule
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
        st.write("### Jauges : Évolution annuelle (2025 vs 2024)")

        # Filtrer les données pour les années 2023 et 2024
        df_filtre = df[df['Année'].isin([2024, 2025])]

        # Groupement par année et calcul des sommes par ville
        df_annuel = df_filtre.groupby(['Année'])[villes].sum().reset_index()

        for ville in villes:
            # Calculer l'évolution en pourcentage de 2024 par rapport à 2023
            consommation_2024 = df_annuel.loc[df_annuel['Année'] == 2024, ville].sum()
            consommation_2025 = df_annuel.loc[df_annuel['Année'] == 2025, ville].sum()

            if consommation_2024 == 0:
                evolution = 0
            else:
                evolution = ((consommation_2025 - consommation_2024) / consommation_2024) * 100

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

    # Fonction : Consommations annuelles de 2025 avec les consommations affichées sur les segments du graphique circulaire
    def tracer_circulaire_2025(df, villes):
        # Vérifier si les colonnes sont présentes
        if not all(v in df.columns for v in villes):
            st.error("Certaines villes ne sont pas présentes dans le DataFrame.")
            return

        # Données 2025
        df_2025 = df[df['Année'] == 2025]
        if df_2025.empty:
            st.warning("Aucune donnée disponible pour 2025.")
            return
        df_2025_sum = df_2025.groupby(['Année'])[villes].sum().reset_index()

        # Données 2024
        df_2024 = df[df['Année'] == 2024]
        if df_2024.empty:
            st.warning("Aucune donnée disponible pour 2024.")
            return
        df_2024_sum = df_2024.groupby(['Année'])[villes].sum().reset_index()

        # Réorganiser les données pour 2025
        df_reshaped_2025 = pd.melt(df_2025_sum, id_vars=['Année'], value_vars=villes)
        df_reshaped_2025.columns = ['Année', 'Ville', 'Consommation']

        # Réorganiser les données pour 2024
        df_reshaped_2024 = pd.melt(df_2024_sum, id_vars=['Année'], value_vars=villes)
        df_reshaped_2024.columns = ['Année', 'Ville', 'Consommation_2024']

        # Fusionner les deux années pour comparer
        df_merged = df_reshaped_2025.merge(df_reshaped_2024[['Ville', 'Consommation_2024']], on='Ville')

        # Calcul des économies
        df_merged['Économie ou Surconsommation'] = df_merged['Consommation_2024'] - df_merged['Consommation']
        df_merged['Coût Evité ou Engentré'] = df_merged['Économie ou Surconsommation'] * 0.18

        # Créer le graphique circulaire
        fig = px.pie(
            df_merged,
            names='Ville',
            values='Consommation',
            title="Consommation annuelle en 2025 pour chaque ville",
            color='Ville',
            hole=0.1
        )

        fig.update_traces(
            textinfo='percent+label',
            pull=[0.1] * len(df_merged),
            textfont=dict(family="Times New Roman")
        )

        fig.update_layout(
            title_font=dict(family="Times New Roman"),
            legend_font=dict(family="Times New Roman")
        )

        # Affichage du graphique
        st.plotly_chart(fig)

        # Préparation du tableau
        df_display = df_merged[['Ville', 'Consommation', 'Économie ou Surconsommation', 'Coût Evité ou Engentré']].copy()
        df_display['Consommation'] = df_display['Consommation'].apply(lambda x: f"{x:,.0f}".replace(",", " ") + " kWh")

        df_display['Économie ou Surconsommation'] = df_merged.apply(
            lambda row: (
                f"<span style='color:{'green' if row['Économie ou Surconsommation'] > 0 else 'red'}'>"
                f"{abs(row['Économie ou Surconsommation']):,.0f}".replace(",", " ") + " kWh  "
                f"<span style='font-weight:bold;color:{'green' if row['Économie ou Surconsommation'] > 0 else 'red'}'>"
                f"{'&#x2B07;' if row['Économie ou Surconsommation'] > 0 else '&#x2B06;'}</span></span>"
            ),
            axis=1
        )

        df_display['Coût Evité ou Engentré'] = df_merged.apply(
            lambda row: (
                f"<span style='color:{'green' if row['Économie ou Surconsommation'] > 0 else 'red'}'>"
                f"{abs(row['Coût Evité ou Engentré']):,.0f}".replace(",", " ") + " €  "
                f"<span style='font-weight:bold;color:{'green' if row['Économie ou Surconsommation'] > 0 else 'red'}'>"
                f"{'&#x2B07;' if row['Économie ou Surconsommation'] > 0 else '&#x2B06;'}</span></span>"
            ),
            axis=1
        )


        # Affichage du tableau sous le graphe
        st.markdown("### Économies par ville entre 2024 et 2025")

        st.markdown(
            f"""
            <style>
                .scrollable-table-wrapper {{
                    max-height: 1200px;
                    overflow-y: auto;
                    margin-top: 10px;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                }}
                .custom-table {{
                    font-family: 'Times New Roman', Times, serif;
                    font-size: 15px;
                    border-collapse: collapse;
                    width: 100%;
                    table-layout: fixed;
                }}
                .custom-table th {{
                    background-color: #f2f2f2;
                    color: DarkGreen;
                    text-align: center;
                    padding: 3px;
                    border: 1px solid #ccc;
                }}
                .custom-table td {{
                    text-align: center;
                    padding: 3px;
                    border: 1px solid #ccc;
                    word-wrap: break-word;
                }}
                .custom-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .custom-table tr:hover {{
                    background-color: #e6f7ff;
                }}
            </style>

            <div class="scrollable-table-wrapper">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Ville</th>
                        <th>Consommations du Trimestre 1 2025</th>
                        <th>Evolution du Trimestres 1 : 2025 vs 2024</th>
                        <th>Evolution du Trimestres 1 : Coût</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join([
                        f"<tr><td>{row['Ville']}</td><td>{row['Consommation']}</td><td>{row['Économie ou Surconsommation']}</td><td>{row['Coût Evité ou Engentré']}</td></tr>"
                        for _, row in df_display.iterrows()
                    ])}
                </tbody>
            </table>
            </div>
            """,
            unsafe_allow_html=True
        )


    # Appel des fonctions

    tracer_circulaire_2025(df, villes)
    tracer_barres_annuelles_avec_annotations(df, villes)
    tracer_barres_mensuelles_2024_avec_annotations(df, villes)
    tracer_barres_trimestrielles_avec_annotations(df, villes)
    # Exemple d'utilisation
    tracer_jauges_annuelles(df, villes)