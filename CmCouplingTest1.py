import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. System Configuration
# ==========================================
# --- Primary Process Models (Standard Oven/Zone Physics) ---
K1, tau11, tau12 = 1.0, 309.2, 1168.8
K2, tau21, tau22 = 1.0, 341.75, 1125.71

# --- DUAL COUPLING PARAMETERS ---
# 1. Flow Stealing (Oil Side): Negative gain, Fast response
K_FLOW_BASE = -0.41  
TAU_FLOW_FAST = 10.0
TAU_FLOW_SLOW = 20.0

# 2. Air Bleed (Thermal Side): Positive gain, Slow response
K_AIR_BASE = 0.2
TAU_AIR_FAST = 400.0
TAU_AIR_SLOW = 800.0

# --- Simulation Settings ---
dt = 2.0
t_end = 20000
baseline_temp = 80.0
sp1, sp2 = 85.0, 75.0
FORCED_ALPHA = 0.6  # This scales both coupling strengths in the sweep

# --- Controller Definitions ---
#PID_FAST = {'Kp': 4.6, 'Ti': 506, 'Td': 126}
#PID_SLOW = {'Kp': 2.4, 'Ti': 2019, 'Td': 504} 
PID_FAST = {'Kp': 2.2, 'Ti': 1168.85, 'Td': 157.37}
#PID_SLOW = {'Kp': 2.2, 'Ti': 1168.85, 'Td': 157.37}
PID_SLOW = {'Kp': 2.2, 'Ti': 1478.03, 'Td': 0.00}

# ==========================================
# 2. Robust PID & Process Classes
# ==========================================
class PID_Robust:
    def __init__(self, Kp, Ti, Td, dt, baseline=0.0):
        self.Kp, self.Ti, self.Td, self.dt = Kp, Ti, Td, dt
        self.integral = (baseline * Ti) / Kp 
        self.prev_err = 0
        
    def compute(self, sp, pv):
        err = sp - pv
        P = self.Kp * err
        D = self.Kp * self.Td * (err - self.prev_err) / self.dt
        
        tentative_integral = self.integral + err * self.dt
        I_tentative = (self.Kp / self.Ti) * tentative_integral
        v = P + I_tentative + D
        
        if v >= 100.0:
            out = 100.0
            if err < 0: self.integral = tentative_integral
        elif v <= 0.0:
            out = 0.0
            if err > 0: self.integral = tentative_integral
        else:
            out = v
            self.integral = tentative_integral
            
        self.prev_err = err
        return out

class SOPDT:
    def __init__(self, K, t1, t2, init_val=0.0):
        self.K, self.t1, self.t2 = K, t1, t2
        self.x1 = init_val
        self.x2 = init_val
        
    def step(self, u):
        dx1 = (-self.x1 + self.K * u) / self.t1
        self.x1 += dx1 * dt
        dx2 = (-self.x2 + self.x1) / self.t2
        self.x2 += dx2 * dt
        return self.x2

# ==========================================
# 3. Simulation Logic (Updated for Dual Coupling)
# ==========================================
def run_simulation(alpha_scale, params1, params2):
    valve1 = PID_Robust(**params1, dt=dt, baseline=baseline_temp)
    valve2 = PID_Robust(**params2, dt=dt, baseline=baseline_temp)
    
    g11 = SOPDT(K1, tau11, tau12, baseline_temp)
    g22 = SOPDT(K2, tau21, tau22, baseline_temp)
    
    # Hydraulic Interaction (Flow Stealing) - Negative
    g_flow12 = SOPDT(K_FLOW_BASE * alpha_scale, TAU_FLOW_SLOW, TAU_FLOW_FAST, 0.0) 
    g_flow21 = SOPDT(K_FLOW_BASE * alpha_scale, TAU_FLOW_SLOW, TAU_FLOW_FAST, 0.0)
    
    # Thermal Interaction (Air Bleed) - Positive
    g_air12 = SOPDT(K_AIR_BASE * alpha_scale, TAU_AIR_SLOW, TAU_AIR_FAST, 0.0)
    g_air21 = SOPDT(K_AIR_BASE * alpha_scale, TAU_AIR_SLOW, TAU_AIR_FAST, 0.0)
    
    pv1, pv2 = baseline_temp, baseline_temp
    h_temp, h_valve = [], []
    
    for _ in range(int(t_end / dt)):
        q1 = valve1.compute(sp1, pv1)
        q2 = valve2.compute(sp2, pv2)
        
        # PV is the sum of: Main Heat + Neighbor Flow Steal + Neighbor Air Bleed
        pv1 = g11.step(q1) + g_flow21.step(q2 - baseline_temp) + g_air21.step(q2 - baseline_temp)
        pv2 = g22.step(q2) + g_flow12.step(q1 - baseline_temp) + g_air12.step(q1 - baseline_temp)
        
        h_temp.append([pv1, pv2])
        h_valve.append([q1, q2])
        
    return np.array(h_temp), np.array(h_valve)

# ==========================================
# 4. Detailed Stability Analysis
# ==========================================
def check_stability_detailed(h_temp, h_valve, tail=1000):
    results = [] 
    MV_EXPECTED_MIDLINE = 80.0
    EPS = 1e-2
    LC_STRAIN_THRESHOLD = (100 - MV_EXPECTED_MIDLINE) - EPS 
    
    for i in range(2):
        pv_hist = h_temp[-tail:, i]
        mv_hist = h_valve[-tail:, i]
        temp_var, valve_var, avg_mv = np.std(pv_hist), np.std(mv_hist), np.mean(mv_hist)
        
        status = "OK"
        if avg_mv > 99.0 or avg_mv < 1.0: status = "SATURATED"
        elif valve_var > 1.0: status = "FIGHTING"
        elif temp_var > 0.1: status = "OSCILLATING"
        elif abs(avg_mv - MV_EXPECTED_MIDLINE) > LC_STRAIN_THRESHOLD: status = "STRAINED"

        results.append({'status': status, 'v_var': valve_var, 't_var': temp_var, 'avg_mv': avg_mv})
    return results[0], results[1]

# ==========================================
# 5. The Sweep & Visualization
# ==========================================
results_data_A, results_data_B = [], []
capture_data = None 

for alpha in np.arange(0.1, 1.1, 0.1):
    rga = 1 / (1 - alpha**2)
    t_A, v_A = run_simulation(alpha, PID_FAST, PID_SLOW)
    P1_A, P2_A = check_stability_detailed(t_A, v_A)
    t_B, v_B = run_simulation(alpha, PID_SLOW, PID_FAST)
    P1_B, P2_B = check_stability_detailed(t_B, v_B)
    
    results_data_A.append([f"{alpha:.1f}", f"{rga:.2f}", P1_A['status'], f"{P1_A['v_var']:.2f}", f"{P1_A['t_var']:.2f}", P2_A['status'], f"{P2_A['v_var']:.2f}", f"{P2_A['t_var']:.2f}"])
    results_data_B.append([f"{alpha:.1f}", f"{rga:.2f}", P1_B['status'], f"{P1_B['v_var']:.2f}", f"{P1_B['t_var']:.2f}", P2_B['status'], f"{P2_B['v_var']:.2f}", f"{P2_B['t_var']:.2f}"])

# Manually capture the requested forced plot
t_forced, v_forced = run_simulation(FORCED_ALPHA, PID_FAST, PID_SLOW)
capture_data = (t_forced, v_forced, FORCED_ALPHA, "Config A - Flow Stealing vs Air Bleed")

# Plotting Logic
t_hist, v_hist, p_alpha, p_title = capture_data
time_ax = np.linspace(0, t_end, len(t_hist))
param_str = (f"Oil Coupling (Neg): $K={K_FLOW_BASE*p_alpha:.2f}, \\tau={TAU_FLOW_FAST}$ | Air Coupling (Pos): $K={K_AIR_BASE*p_alpha:.2f}, \\tau={TAU_AIR_FAST}$")

fig = plt.figure(figsize=(14, 15)) 
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 2])

for idx, (data, title) in enumerate([(results_data_A, "Config A: Standard"), (results_data_B, "Config B: Swapped")]):
    ax = fig.add_subplot(gs[idx])
    ax.axis('off'); ax.axis('tight')
    tbl = ax.table(cellText=data, colLabels=("Alpha", "RGA", "P1 Stat", "P1 MV Var", "P1 T Var", "P2 Stat", "P2 MV Var", "P2 T Var"), loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    ax.set_title(title, fontweight='bold')

ax_plt = fig.add_subplot(gs[2])
ax_plt.plot(time_ax, t_hist[:, 0], 'r-', label='Temp 1 (PV)')
ax_plt.plot(time_ax, t_hist[:, 1], 'b-', label='Temp 2 (PV)')
ax_plt.plot(time_ax, v_hist[:, 0], 'r--', alpha=0.5, label='Valve 1 (MV)')
ax_plt.plot(time_ax, v_hist[:, 1], 'b--', alpha=0.5, label='Valve 2 (MV)')
ax_plt.set_title(f"Dynamic Analysis: {p_title} (Alpha Scale = {p_alpha:.2f})", pad=40, fontweight='bold')
ax_plt.text(0.5, 1.03, param_str, transform=ax_plt.transAxes, ha='center', fontsize=9, color='gray') 
ax_plt.legend(loc='upper right'); ax_plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
