import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="RTD-Simulation", layout="wide")
st.title("🧪 RTD-Simulation – Verweilzeitverteilung")

# Sidebar – Steuerung
st.sidebar.header("🔧 Simulationsparameter")

modelle = st.sidebar.multiselect(
    "Reaktormodell auswählen",
    ["Axial Dispersion", "Tanks-in-Series", "Plug Flow (PFTR)", "CSTR"],
    default=["Tanks-in-Series"]
)

t_max = st.sidebar.slider("Maximale Zeit (t)", 5.0, 100.0, 30.0)
t_points = st.sidebar.slider("Anzahl Zeitpunkte", 100, 2000, 500)
t = np.linspace(0.001, t_max, int(t_points))
dt = t[1] - t[0]

anzeige = st.sidebar.radio("Was soll dargestellt werden?", ["Nur E(t)", "Nur F(t)", "Beides"])
log_y = st.sidebar.checkbox("Logarithmische Y-Achse")
dimensionless_time = st.sidebar.checkbox("Dimensionslose Zeit verwenden (θ = t / τ)")

# Plot vorbereiten
fig, ax1 = plt.subplots(figsize=(9, 6))
color_index = 0
farben = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

daten_liste = []

# Zweite Y-Achse für F(t)
ax2 = ax1.twinx() if anzeige == "Beides" else None

for modell in modelle:
    farbe = farben[color_index % len(farben)]
    color_index += 1

    if modell == "Axial Dispersion":
        Bo = st.sidebar.slider("Bodenstein-Zahl (Bo)", 1.0, 1000.0, 20.0, step=1.0)
        E = (1 / np.sqrt(4 * np.pi / Bo * t)) * np.exp(-(1 - t)**2 * Bo / (4 * t))
        beschriftung = f"Axial Dispersion (Bo={Bo:.0f})"

    elif modell == "Tanks-in-Series":
        N = st.sidebar.slider("Anzahl CSTRs (N)", 1, 100, 5)
        E = (N**N) * t**(N - 1) * np.exp(-N * t) / np.math.factorial(N - 1)
        beschriftung = f"Tanks-in-Series (N={N})"

    elif modell == "Plug Flow (PFTR)":
        E = np.zeros_like(t)
        E[np.argmin(np.abs(t - 1))] = 1 / dt  # Dirac-Delta-Näherung
        beschriftung = "Plug Flow (PFTR)"

    elif modell == "CSTR":
        E = np.exp(-t)
        beschriftung = "CSTR"

    F = np.cumsum(E) * dt
    tau = np.trapz(t * E, t)
    theta = t / tau  # dimensionslose Zeit

    df = pd.DataFrame({
        "Zeit (t)": t,
        "θ (t/τ)": theta,
        "E(t)": E,
        "F(t)": F,
        "Modell": beschriftung,
        "Verweilzeit (τ)": tau
    })
    daten_liste.append(df)

    # X-Achse je nach Einstellung
    x_axis = theta if dimensionless_time else t

    # Plotten
    if anzeige == "Nur E(t)":
        ax1.plot(x_axis, E, label=f"E(t) – {beschriftung}", color=farbe)
    elif anzeige == "Nur F(t)":
        ax1.plot(x_axis, F, label=f"F(t) – {beschriftung}", color=farbe)
    elif anzeige == "Beides":
        ax1.plot(x_axis, E, label=f"E(t) – {beschriftung}", color=farbe, linestyle="-")
        ax2.plot(x_axis, F, label=f"F(t) – {beschriftung}", color=farbe, linestyle="--")

    # Ausgabe mittlere Verweilzeit
    st.markdown(f"**{beschriftung}** — Mittlere Verweilzeit (τ): `{tau:.3f}`")

# Achsentitel und Layout
ax1.set_xlabel("θ = t / τ" if dimensionless_time else "Zeit (t)")
if anzeige != "Nur F(t)":
    ax1.set_ylabel("E(t)")
    if log_y:
        ax1.set_yscale("log")
if anzeige == "Nur F(t)":
    ax1.set_ylabel("F(t)")
    if log_y:
        ax1.set_yscale("log")
if anzeige == "Beides":
    ax2.set_ylabel("F(t)")
    if log_y:
        ax1.set_yscale("log")
        ax2.set_yscale("log")

ax1.grid(True)

# Legende zusammenführen bei Beides
if anzeige == "Beides":
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)
else:
    ax1.legend()

st.pyplot(fig)

# Download-Bereich
combined_df = pd.concat(daten_liste, ignore_index=True)
csv = combined_df.to_csv(index=False).encode()
st.download_button("⬇️ CSV herunterladen", data=csv, file_name="RTD_Simulation.csv", mime="text/csv")

buf = BytesIO()
fig.savefig(buf, format="png")
buf.seek(0)
st.download_button("🖼️ Diagramm als PNG speichern", data=buf, file_name="RTD_Plot.png", mime="image/png")
