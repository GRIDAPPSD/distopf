def admm_boundary_penalty(m, rho: float = 1.0):
    """Quadratic penalty toward ADMM target values stored in schedules.

    - Swing voltages target 'v_a', 'v_b', 'v_c'.
    - Dummy boundary loads target '<area>.<phase>.p' and '<area>.<phase>.q'.
    """
    penalty = 0

    # Voltage penalty at swing buses.
    if hasattr(m, "swing_phase_set"):
        for _id, ph in m.swing_phase_set:
            for t in m.time_set:
                v_target_sq = m.v_swing[_id, ph, t] ** 2
                penalty += (m.v2[_id, ph, t] - v_target_sq) ** 2

    # Power penalty at dummy boundary loads: any bus whose load_shape refers
    # to a schedule column that supplies '<area>.<phase>.p' or '.q' should be
    # treated as an ADMM boundary rather than a fixed load.
    for _id, ph in m.bus_phase_set:
        for t in m.time_set:
            p_param_name = f"schedule_{m.name_map[_id]}.{ph}.p"
            q_param_name = f"schedule_{m.name_map[_id]}.{ph}.q"

            if hasattr(m, p_param_name) and hasattr(m, q_param_name):
                p_target = getattr(m, p_param_name)[t]
                q_target = getattr(m, q_param_name)[t]

                penalty += (m.p_load[_id, ph, t] - p_target) ** 2
                penalty += (m.q_load[_id, ph, t] - q_target) ** 2

    return 0.5 * rho * penalty
