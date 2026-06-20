"""
Battery Management System (BMS) Simulator
==========================================
Features:
 1. Extended Kalman Filter (EKF) SOC estimation (1-RC Thevenin equivalent circuit model)
 2. SOH (State of Health) estimation via Ah-throughput capacity fade
 3. Thermal model (lumped-mass heat generation + convective dissipation)
 4. Fault detection: over-voltage, under-voltage, over-temperature, over-current
 5. Charging mode simulation: CC-CV (constant current / constant voltage) and discharge
 6. Streamlit dashboard with live-updating plots

Run with:
    streamlit run bms_simulation.py
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# 1. BATTERY ELECTRICAL MODEL (1-RC Thevenin equivalent circuit)
# ----------------------------------------------------------------------

def ocv_from_soc(soc: float) -> float:
    """Open-circuit-voltage vs SOC curve (typical Li-ion single cell, 0-1 SOC)."""
    soc = np.clip(soc, 0.0, 1.0)
    return 3.2 + 0.6 * soc + 0.5 * soc**2 - 0.3 * soc**3


def docv_dsoc(soc: float) -> float:
    """Derivative of OCV curve wrt SOC (needed for EKF Jacobian)."""
    soc = np.clip(soc, 0.0, 1.0)
    return 0.6 + 1.0 * soc - 0.9 * soc**2


class BatteryPlant:
    """The 'real' battery being simulated (ground truth), with degradation."""

    def __init__(self, capacity_ah=3.0, r0=0.05, r1=0.03, c1=2000.0):
        self.capacity_ah_init = capacity_ah   # nameplate capacity
        self.soh = 100.0                      # %
        self.r0 = r0                          # ohmic resistance
        self.r1 = r1                          # polarization resistance
        self.c1 = c1                          # polarization capacitance (F)
        self.tau = r1 * c1

        self.soc_true = 0.5                   # ground-truth SOC (0-1)
        self.v1 = 0.0                          # voltage across RC branch
        self.temp = 25.0                       # deg C
        self.ah_throughput = 0.0               # cumulative |I|*dt for SOH fade

        # Thermal parameters
        self.thermal_mass = 60.0    # J/°C  (lumped thermal capacitance)
        self.h_conv = 0.6           # W/°C  (heat transfer coefficient to ambient)

    @property
    def capacity_ah(self):
        return self.capacity_ah_init * (self.soh / 100.0)

    def step(self, current_a: float, dt: float, t_ambient: float):
        """Advance the true battery state by dt seconds.
        current_a > 0 = discharge, current_a < 0 = charge (convention used throughout)."""

        # --- Electrical: SOC via coulomb counting (ground truth) ---
        self.soc_true -= (current_a * dt) / (3600.0 * self.capacity_ah)
        self.soc_true = np.clip(self.soc_true, 0.0, 1.0)

        # --- RC branch dynamics ---
        self.v1 = self.v1 * np.exp(-dt / self.tau) + self.r1 * (1 - np.exp(-dt / self.tau)) * current_a

        # --- Terminal voltage ---
        v_terminal = ocv_from_soc(self.soc_true) - current_a * self.r0 - self.v1

        # --- Thermal model: I^2*R heat generation, convective loss to ambient ---
        heat_gen = (current_a ** 2) * self.r0      # Watts
        heat_loss = self.h_conv * (self.temp - t_ambient)
        dT = (heat_gen - heat_loss) / self.thermal_mass * dt
        self.temp += dT

        # --- SOH fade: capacity loss proportional to Ah throughput ---
        self.ah_throughput += abs(current_a) * dt / 3600.0
        # Lose ~0.002% SOH per Ah throughput (tunable degradation rate)
        self.soh = max(70.0, 100.0 - 0.002 * self.ah_throughput)

        return v_terminal


# ----------------------------------------------------------------------
# 2. EXTENDED KALMAN FILTER FOR SOC ESTIMATION
# ----------------------------------------------------------------------

class SOC_EKF:
    """Estimates SOC from noisy current/voltage measurements using an EKF
    over the same 1-RC Thevenin model used by the plant."""

    def __init__(self, capacity_ah, r0, r1, c1, soc_init=0.5):
        self.capacity_ah = capacity_ah
        self.r0 = r0
        self.r1 = r1
        self.c1 = c1
        self.tau = r1 * c1

        self.x = np.array([soc_init, 0.0])          # state: [SOC, V1]
        self.P = np.diag([1e-3, 1e-3])               # state covariance
        self.Q = np.diag([1e-7, 1e-6])                # process noise
        self.R = 4e-4                                 # measurement (voltage) noise variance

    def predict(self, current_a, dt):
        soc, v1 = self.x
        a = np.exp(-dt / self.tau)

        soc_pred = soc - (current_a * dt) / (3600.0 * self.capacity_ah)
        v1_pred = a * v1 + self.r1 * (1 - a) * current_a
        self.x = np.array([np.clip(soc_pred, 0.0, 1.0), v1_pred])

        F = np.array([[1.0, 0.0],
                      [0.0, a]])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, v_meas, current_a):
        soc, v1 = self.x
        v_pred = ocv_from_soc(soc) - current_a * self.r0 - v1
        y = v_meas - v_pred  # innovation

        H = np.array([docv_dsoc(soc), -1.0])
        S = H @ self.P @ H.T + self.R
        K = (self.P @ H) / S

        self.x = self.x + K * y
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P

    def step(self, current_a, dt, v_meas):
        self.predict(current_a, dt)
        self.update(v_meas, current_a)
        return self.x[0]  # SOC estimate


# ----------------------------------------------------------------------
# 3. FAULT DETECTION
# ----------------------------------------------------------------------

FAULT_LIMITS = dict(
    v_over=4.25,
    v_under=2.8,
    t_over=55.0,
    i_over=8.0,   # amps
)


def detect_faults(voltage, temp, current):
    faults = []
    if voltage > FAULT_LIMITS["v_over"]:
        faults.append(f"OVER-VOLTAGE ({voltage:.2f} V)")
    if voltage < FAULT_LIMITS["v_under"]:
        faults.append(f"UNDER-VOLTAGE ({voltage:.2f} V)")
    if temp > FAULT_LIMITS["t_over"]:
        faults.append(f"OVER-TEMPERATURE ({temp:.1f} °C)")
    if abs(current) > FAULT_LIMITS["i_over"]:
        faults.append(f"OVER-CURRENT ({current:.2f} A)")
    return faults


# ----------------------------------------------------------------------
# 4. CHARGING MODE (CC-CV) / DISCHARGE CURRENT PROFILE
# ----------------------------------------------------------------------

def get_current(mode, plant, cc_current, cv_voltage, discharge_current):
    """Returns the current command for this step.
    Sign convention: positive = discharge, negative = charge."""
    if mode == "Charging (CC-CV)":
        v_now = ocv_from_soc(plant.soc_true) - plant.v1  # approx terminal V at I=0
        if v_now < cv_voltage:
            return -cc_current          # constant current phase (charging => negative)
        else:
            # constant voltage phase: taper current as it approaches full
            taper = max(0.02, (1.0 - plant.soc_true)) * cc_current
            return -min(cc_current, taper)
    elif mode == "Discharging":
        return discharge_current
    else:  # Idle
        return 0.0


# ----------------------------------------------------------------------
# 5. STREAMLIT DASHBOARD
# ----------------------------------------------------------------------

st.set_page_config(page_title="BMS Simulator", layout="wide")
st.title("Battery Management System (BMS) Simulator")
st.caption("EKF SOC estimation • SOH tracking • Thermal model • Fault detection • CC-CV charging")

# ---- Sidebar controls ----
st.sidebar.header("Simulation Settings")
mode = st.sidebar.selectbox("Mode", ["Discharging", "Charging (CC-CV)", "Idle"])
cc_current = st.sidebar.slider("Charge CC current (A)", 0.5, 5.0, 2.0, 0.1)
cv_voltage = st.sidebar.slider("Charge CV target voltage (V)", 3.8, 4.2, 4.1, 0.01)
discharge_current = st.sidebar.slider("Discharge current (A)", 0.5, 8.0, 2.0, 0.1)
t_ambient = st.sidebar.slider("Ambient temperature (°C)", -10, 45, 25)
dt = st.sidebar.slider("Time step (s)", 0.5, 5.0, 1.0, 0.5)
n_steps = st.sidebar.slider("Steps per run", 50, 1000, 300, 50)
noise_std = st.sidebar.slider("Voltage sensor noise (std, V)", 0.0, 0.05, 0.01, 0.005)
col_left, col_right = st.columns([8, 2])

with col_right:
    run_button = st.button(
        "▶ Run Simulation",
        use_container_width=True
    )

with col_left:
    reset_button = st.button(
        "⟲ Reset Battery State"
    )

# ---- Persistent state across reruns ----
if "plant" not in st.session_state or reset_button:
    st.session_state.plant = BatteryPlant()
    st.session_state.ekf = SOC_EKF(
        capacity_ah=st.session_state.plant.capacity_ah,
        r0=st.session_state.plant.r0,
        r1=st.session_state.plant.r1,
        c1=st.session_state.plant.c1,
        soc_init=st.session_state.plant.soc_true,
    )
    st.session_state.history = pd.DataFrame(columns=[
        "t", "SOC_true", "SOC_ekf", "Voltage", "Current", "Temp", "SOH"
    ])
    st.session_state.t = 0.0
    st.session_state.faults_log = []

plant = st.session_state.plant
ekf = st.session_state.ekf

# ---- Live status placeholders ----
col1, col2, col3, col4 = st.columns(4)
soc_box = col1.empty()
soh_box = col2.empty()
temp_box = col3.empty()
volt_box = col4.empty()

fault_box = st.empty()
chart_soc = st.empty()
chart_voltage = st.empty()
chart_temp = st.empty()
table_box = st.empty()


def render(latest_row, faults):
    soc_box.metric("SOC (EKF estimate)", f"{latest_row['SOC_ekf']*100:.1f} %",
                    f"true: {latest_row['SOC_true']*100:.1f} %")
    soh_box.metric("SOH", f"{latest_row['SOH']:.2f} %")
    temp_box.metric("Temperature", f"{latest_row['Temp']:.1f} °C")
    volt_box.metric("Terminal Voltage", f"{latest_row['Voltage']:.3f} V")

    if faults:
        fault_box.error(" | ".join(faults))
    else:
        fault_box.success("No faults detected ✅")

    hist = st.session_state.history
    fig_soc = go.Figure()
    fig_soc.add_trace(go.Scatter(x=hist["t"], y=hist["SOC_true"] * 100, name="SOC (true)"))
    fig_soc.add_trace(go.Scatter(x=hist["t"], y=hist["SOC_ekf"] * 100, name="SOC (EKF est.)", line=dict(dash="dot")))
    fig_soc.update_layout(title="State of Charge", xaxis_title="Time (s)", yaxis_title="SOC (%)", height=300)
    chart_soc.plotly_chart(fig_soc, use_container_width=True, key=f"soc_{len(hist)}")

    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=hist["t"], y=hist["Voltage"], name="Terminal Voltage"))
    fig_v.add_hline(y=FAULT_LIMITS["v_over"], line_dash="dash", line_color="red", annotation_text="Over-V limit")
    fig_v.add_hline(y=FAULT_LIMITS["v_under"], line_dash="dash", line_color="orange", annotation_text="Under-V limit")
    fig_v.update_layout(title="Terminal Voltage", xaxis_title="Time (s)", yaxis_title="Volts", height=300)
    chart_voltage.plotly_chart(fig_v, use_container_width=True, key=f"v_{len(hist)}")

    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=hist["t"], y=hist["Temp"], name="Temperature"))
    fig_t.add_hline(y=FAULT_LIMITS["t_over"], line_dash="dash", line_color="red", annotation_text="Over-temp limit")
    fig_t.update_layout(title="Battery Temperature", xaxis_title="Time (s)", yaxis_title="°C", height=300)
    chart_temp.plotly_chart(fig_t, use_container_width=True, key=f"t_{len(hist)}")

    table_box.dataframe(hist.tail(10).iloc[::-1], use_container_width=True)


# ---- Main simulation loop ----
if run_button:
    for _ in range(n_steps):
        current = get_current(mode, plant, cc_current, cv_voltage, discharge_current)

        v_true = plant.step(current, dt, t_ambient)
        v_meas = v_true + np.random.normal(0, noise_std)  # noisy sensor reading

        soc_est = ekf.step(current, dt, v_meas)
        ekf.capacity_ah = plant.capacity_ah  # EKF tracks fading capacity from SOH block

        faults = detect_faults(v_meas, plant.temp, current)
        if faults:
            st.session_state.faults_log.append((st.session_state.t, faults))

        st.session_state.t += dt
        new_row = {
            "t": st.session_state.t,
            "SOC_true": plant.soc_true,
            "SOC_ekf": soc_est,
            "Voltage": v_meas,
            "Current": current,
            "Temp": plant.temp,
            "SOH": plant.soh,
        }
        st.session_state.history = pd.concat(
            [st.session_state.history, pd.DataFrame([new_row])], ignore_index=True
        )

        render(new_row, faults)
        time.sleep(0.02)  # small pause so charts visibly animate

    st.success(f"Simulation complete — {n_steps} steps, {n_steps*dt:.0f} s simulated.")

elif len(st.session_state.history) > 0:
    last = st.session_state.history.iloc[-1].to_dict()
    last_faults = detect_faults(last["Voltage"], last["Temp"], last["Current"])
    render(last, last_faults)
else:
    st.info("Set parameters in the sidebar and click **Run Simulation** to start.")
