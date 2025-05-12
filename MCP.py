import streamlit as st
import numpy as np
import plotly.graph_objs as go

# Titre de l'application
st.title("Modélisation de l'accumulation de chaleur avec des matériaux à changement de phase (MCP)")

# --- Entrée des paramètres utilisateur ---
st.sidebar.header("Paramètres du matériau")
mat_type = st.sidebar.selectbox("Type de matériau", ["Standard", "Avec MCP"])

# Paramètres de simulation
duration = st.sidebar.slider("Durée de la simulation (heures)", 1, 48, 24)
dt = 60  # pas de temps en secondes
time_steps = int(duration * 3600 / dt)

# Température extérieure simulée (sinusoïdale simplifiée)
t = np.linspace(0, duration * 3600, time_steps)
T_ext = 15 + 10 * np.sin(2 * np.pi * t / (24 * 3600))  # variation réaliste autour de 15-25°C

# Paramètres thermiques
st.sidebar.subheader("Propriétés thermiques")
if mat_type == "Standard":
    rho = 800       # densité kg/m3
    c = 1000         # capacité thermique J/kg.K
    k = 0.2          # conductivité thermique W/m.K
    L = 0            # chaleur latente J/kg
    T_melt = None
else:
    rho = st.sidebar.number_input("Densité (kg/m3)", min_value=1.0, value=800.0)
    c = st.sidebar.number_input("Capacité thermique (J/kg.K)", min_value=1.0, value=2000.0)
    k = st.sidebar.number_input("Conductivité thermique (W/m.K)", min_value=1e-6, value=0.2)
    L = st.sidebar.number_input("Chaleur latente (J/kg)", min_value=0.0, value=200000.0)
    T_melt = st.sidebar.slider("Température de changement de phase (°C)", 15, 30, 25)

# Conditions initiales et géométrie
length = 0.1  # m
nx = 50
x = np.linspace(0, length, nx)
dx = length / (nx - 1)
T_init = 20  # °C

# Initialisation des températures
T = np.ones(nx) * T_init
T_record = []

# --- Simulation thermique simple ---
alpha = k / (rho * c)
if not np.isfinite(alpha) or alpha <= 0:
    st.error("Paramètres physiques invalides : alpha (diffusivité thermique) non fini ou négatif.")
else:
    for i in range(time_steps):
        T_new = T.copy()
        for j in range(1, nx - 1):
            T_new[j] = T[j] + alpha * dt / dx**2 * (T[j+1] - 2*T[j] + T[j-1])

            # Gestion du MCP pendant le changement de phase
            if mat_type == "Avec MCP" and T_melt - 0.5 < T[j] < T_melt + 0.5:
                T_new[j] += L / (c * 10)  # effet simplifié du MCP (phase tampon thermique)

            # Borne de sécurité
            T_new[j] = np.clip(T_new[j], -50, 100)

        # Conditions aux limites (Dirichlet)
        T_new[0] = T_ext[i]
        T_new[-1] = T[-2]  # isolement thermique à droite

        T = T_new.copy()
        T_record.append(T.copy())

T_record = np.array(T_record)

# --- Affichage des résultats ---
st.subheader("Évolution de la température dans le matériau")

if T_record.size == 0 or np.isnan(T_record).any() or np.isinf(T_record).any():
    st.error("Erreur : les données de température sont invalides (NaN, Inf ou vides).")
else:
    time_hours = np.linspace(0, duration, len(T_record))
    z = x  # profondeur

    # Préparation des données pour plotly
    temps = np.tile(time_hours, (nx, 1)).T
    profondeur = np.tile(z, (len(time_hours), 1))
    temp_data = T_record

    fig_temp = go.Figure(data=go.Heatmap(
        z=temp_data.T,
        x=time_hours,
        y=z,
        colorscale='Inferno',
        colorbar=dict(title='Température (°C)')
    ))
    fig_temp.update_layout(
        xaxis_title="Temps (h)",
        yaxis_title="Profondeur (m)",
        title="Distribution de la température dans le matériau"
    )
    st.plotly_chart(fig_temp)

    # Quantité de chaleur stockée (approximation)
    dQ = rho * c * np.trapz((T_record - T_init), dx=dt, axis=0)
    Q_total = np.sum(dQ) / 1000  # en kJ

    st.subheader("Analyse énergétique")
    st.write(f"Chaleur totale accumulée sur {duration} h : **{Q_total:.2f} kJ**")

# Affichage de la température extérieure
st.subheader("Température extérieure")
fig_ext = go.Figure()
fig_ext.add_trace(go.Scatter(x=t / 3600, y=T_ext, mode='lines', name='T_ext'))
fig_ext.update_layout(
    xaxis_title="Temps (h)",
    yaxis_title="Température (°C)",
    title="Température extérieure",
    template="plotly_white"
)
st.plotly_chart(fig_ext)

st.markdown("---")
st.markdown("Application développée pour explorer l'impact des MCP dans la construction durable.")
