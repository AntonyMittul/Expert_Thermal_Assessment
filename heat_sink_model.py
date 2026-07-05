"""
Heat sink thermal model - refactored from the base script provided by
Expert Thermal/XThermal so that the key operating parameters (TDP, air
velocity, TIM conductivity) can be passed in as arguments.

The physics is kept identical to the original script:
junction -> case (R_jc) -> TIM (R_tim) -> heat sink convection (R_conv) -> ambient
"""

# ---- Fixed geometry (from the original script / reference report) ----
L_DIE = 0.0525          # die length, m
W_DIE = 0.045           # die width, m

L = 90e-3               # heat sink length, m
W = 116e-3              # heat sink width, m
H = 27e-3               # heat sink overall height, m
FIN_THICKNESS = 0.8e-3  # m
N_FINS = 60
BASE_THICKNESS = 2.5e-3 # m

T_TIM = 0.1e-3          # TIM thickness, m
R_JC = 0.2              # junction-to-case resistance, degC/W

# ---- Air properties at 25 degC ----
K_AIR = 0.0262          # W/m.K
NU_AIR = 1.57e-5        # m^2/s
PR_AIR = 0.71


def compute_thermals(TDP, V_air, k_tim, T_ambient=25.0):
    """Run the thermal resistance network for one operating point.

    Returns a dict with the intermediate quantities and the two outputs
    we care about: total thermal resistance and junction temperature.
    """
    # fin spacing (channel width between adjacent fins)
    s_f = (W - (N_FINS * FIN_THICKNESS)) / (N_FINS - 1)

    # Reynolds number based on fin spacing as characteristic length
    Re = (V_air * s_f) / NU_AIR

    # Nusselt number: Sieder-Tate for laminar, Dittus-Boelter for turbulent
    if Re < 2300:
        Nu = 1.86 * ((Re * PR_AIR * (2 * s_f) / L) ** (1 / 3))
    else:
        Nu = 0.023 * (Re ** 0.8) * (PR_AIR ** 0.3)

    # convective heat transfer coefficient (hydraulic diameter ~ 2*s_f)
    h = (Nu * K_AIR) / (2 * s_f)

    # convective surface area: fins + exposed base between fins
    h_fin = H - BASE_THICKNESS
    A_fin = N_FINS * (2 * h_fin * L) + (s_f * L)
    A_base_convection = (L * W) - (FIN_THICKNESS * N_FINS * L)
    A_total = A_fin + A_base_convection

    R_conv = 1 / (h * A_total)

    # TIM resistance over the die footprint
    A_die = L_DIE * W_DIE
    R_tim = T_TIM / (k_tim * A_die)

    R_hs = R_conv + R_tim
    R_total = R_JC + R_hs

    T_j = T_ambient + TDP * R_total

    return {
        "TDP_W": TDP,
        "V_air_m_s": V_air,
        "k_tim_W_mK": k_tim,
        "fin_spacing_m": s_f,
        "Re": Re,
        "Nu": Nu,
        "h_W_m2K": h,
        "R_conv_C_W": R_conv,
        "R_tim_C_W": R_tim,
        "R_total_C_W": R_total,
        "T_junction_C": T_j,
    }


if __name__ == "__main__":
    # sanity check against the original script's default case
    result = compute_thermals(TDP=150, V_air=1.0, k_tim=4.0)
    for key, value in result.items():
        print(f"{key}: {value:.6g}")
