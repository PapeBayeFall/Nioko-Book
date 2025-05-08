import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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
T_ext = 10 + 10 * np.sin(2 * np.pi * t / (24 * 3600))  # variation sur 24h

# Paramètres thermiques
st.sidebar.subheader("Propriétés thermiques")
if mat_type == "Standard":
    rho = 800       # densité kg/m3
    c = 1000         # capacité thermique J/kg.K
    k = 0.2          # conductivité thermique W/m.K
    L = 0            # chaleur latente J/kg
    T_melt = None
else:
    rho = st.sidebar.number_input("Densité (kg/m3)", 800)
    c = st.sidebar.number_input("Capacité thermique (J/kg.K)", 2000)
    k = st.sidebar.number_input("Conductivité thermique (W/m.K)", 0.2)
    L = st.sidebar.number_input("Chaleur latente (J/kg)", 200000)
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
for i in range(time_steps):
    T_new = T.copy()
    for j in range(1, nx - 1):
        alpha = k / (rho * c)
        T_new[j] = T[j] + alpha * dt / dx**2 * (T[j+1] - 2*T[j] + T[j-1])

        # Gestion du MCP pendant le changement de phase
        if mat_type == "Avec MCP" and T_melt - 0.5 < T[j] < T_melt + 0.5:
            T_new[j] += L / (c * 10)  # effet simplifié du MCP (phase tampon thermique)

    # Conditions aux limites (Dirichlet)
    T_new[0] = T_ext[i]
    T_new[-1] = T[-2]  # isolement thermique à droite

    T = T_new.copy()
    T_record.append(T.copy())

T_record = np.array(T_record)

# --- Affichage des résultats ---
st.subheader("Évolution de la température dans le matériau")

fig, ax = plt.subplots(figsize=(10, 4))
img = ax.imshow(T_record.T, aspect='auto', extent=[0, duration, 0, length], origin='lower', cmap='inferno')
fig.colorbar(img, ax=ax, label="Température (°C)")
ax.set_xlabel("Temps (heures)")
ax.set_ylabel("Profondeur (m)")
st.pyplot(fig)

# Quantité de chaleur stockée (approximation)
dQ = rho * c * np.trapz((T_record - T_init), dx=dt, axis=0)
Q_total = np.sum(dQ) / 1000  # en kJ

st.subheader("Analyse énergétique")
st.write(f"Chaleur totale accumulée sur {duration} h : **{Q_total:.2f} kJ**")

# Affichage de la température extérieure
st.subheader("Température extérieure")
fig2, ax2 = plt.subplots()
ax2.plot(t / 3600, T_ext, label="T_ext")
ax2.set_xlabel("Temps (h)")
ax2.set_ylabel("Température (°C)")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

st.markdown("---")
st.markdown("Application développée pour explorer l'impact des MCP dans la construction durable.")
