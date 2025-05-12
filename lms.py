import streamlit as st

# Dictionnaire de la structure LMS
menu_structure = {
    "Gouvernance et Pilotage": [
        "Politique de formation",
        "Plan stratégique",
        "Comité de pilotage",
        "Tableaux de bord de performance"
    ],
    "Gestion des Formations": [
        "Catalogue des formations",
        "Création de sessions",
        "Suivi des inscriptions",
        "Évaluation des sessions"
    ],
    "Gestion des Utilisateurs": [
        "Apprenants",
        "Formateurs",
        "Administrateurs",
        "Rôles et permissions"
    ],
    "Suivi et Évaluation": [
        "Quiz et examens",
        "Résultats et statistiques",
        "Feedback des apprenants",
        "Tableaux de suivi des progrès"
    ],
    "Contenus Pédagogiques": [
        "Modules SCORM / Vidéos",
        "Documents PDF / PPT",
        "Cours interactifs",
        "Banque de ressources"
    ],
    "Communication et Collaboration": [
        "Messagerie interne",
        "Annonces",
        "Forums",
        "Groupes de travail"
    ],
    "Gestion Administrative": [
        "Facturation",
        "Certificats",
        "Contrats formateurs",
        "Rapports réglementaires"
    ],
    "Statistiques et Reporting": [
        "Indicateurs de performance",
        "Export Excel / PDF",
        "Rapports personnalisés",
        "Visualisation graphique"
    ],
    "Paramètres du Système": [
        "Personnalisation de l’interface",
        "Gestion des accès",
        "Langues",
        "Sauvegardes & restaurations"
    ],
    "Archives": [
        "Anciens modules",
        "Historique des formations",
        "Logs utilisateurs",
        "Anciennes statistiques"
    ]
}

st.set_page_config(page_title="LMS Interface", layout="wide")

st.markdown("""
    <style>
    .menu-container {
        display: flex;
        justify-content: flex-start;
        gap: 2rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .menu-selected {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 4px;
    }
    .menu-unselected {
        color: black;
        padding-bottom: 4px;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📚 Interface LMS - Tableau de Bord")

if "selected_menu" not in st.session_state:
    st.session_state.selected_menu = list(menu_structure.keys())[0]

st.markdown('<div class="menu-container">', unsafe_allow_html=True)
cols = st.columns(len(menu_structure))

for i, (col, menu) in enumerate(zip(cols, menu_structure.keys())):
    with col:
        if st.button(menu, key=f"btn_{i}"):
            st.session_state.selected_menu = menu

st.markdown('</div>', unsafe_allow_html=True)

# Affichage des sous-menus
selected_menu = st.session_state.selected_menu
st.subheader(f"📂 Sous-menus de : {selected_menu}")

for submenu in menu_structure[selected_menu]:
    with st.expander(f"📁 {submenu}"):
        if selected_menu == "Gestion des Formations":
            if submenu == "Catalogue des formations":
                st.write("Liste des formations disponibles :")
                st.table({"Titre": ["Python avancé", "Introduction à la cybersécurité"], "Durée": ["3 jours", "2 jours"]})

            elif submenu == "Création de sessions":
                st.write("Créer une nouvelle session de formation :")
                with st.form("session_form"):
                    nom = st.text_input("Nom de la session")
                    date = st.date_input("Date")
                    formateur = st.text_input("Nom du formateur")
                    submitted = st.form_submit_button("Créer")
                    if submitted:
                        st.success(f"Session '{nom}' créée pour le {date} avec le formateur {formateur}.")

            elif submenu == "Suivi des inscriptions":
                st.write("Suivi des inscriptions aux sessions en cours")
                st.dataframe({"Nom": ["Alice", "Bob"], "Session": ["Python avancé", "Cybersécurité"]})

            elif submenu == "Évaluation des sessions":
                st.write("Rapports d'évaluation :")
                st.write("- Session Python : 4.5/5")
                st.write("- Session Cybersécurité : 4.2/5")
        else:
            st.write(f"Contenu à implémenter pour la section **{submenu}**")
