import plotly.express as px
import pandas as pd
import numpy as np
import distopf as opf


test2x1 = opf.CASES_DIR / "dss/2Bus_1phase/2Bus1ph.DSS"
test2 = opf.CASES_DIR / "dss/2Bus/2Bus.DSS"
test2D = opf.CASES_DIR / "dss/2BusD/2Bus.DSS"
test3 = opf.CASES_DIR / "dss/3Bus/3Bus.DSS"
ieee4 = opf.CASES_DIR / "dss/4Bus-YY-Bal/4Bus-YY-Bal.DSS"
ieee4YD = opf.CASES_DIR / "dss/4Bus-YD-Bal/4Bus-YD-Bal.DSS"
ieee13 = opf.CASES_DIR / "dss/ieee13_dss/IEEE13Nodeckt.dss"
ieee34 = opf.CASES_DIR / "dss/34Bus/Run_IEEE34Mod2.dss"
ieee123 = opf.CASES_DIR / "dss/ieee123_dss/Run_IEEE123Bus.DSS"
rahul123 = opf.CASES_DIR / "dss/rahul123/ieee123master_base.dss"
ieee_9500 = opf.CASES_DIR / "dss/9500-primary-network/Master.dss"
dirs = [
    # test2,
    # test2D,
    # test3,
    # ieee4,
    # ieee4YD,
    # ieee13,
    # ieee34,
    # ieee123,
    rahul123,
    # ieee_9500,
]
names = {
    # "ieee13_dss": "IEEE 13-Bus",
    # "ieee123_dss": "IEEE 123-Bus",
    "rahul123": "Rahul's 123-Bus",
    # "9500-primary-network": "IEEE 9500 Primary Network",
}
# ~~~~~ Rahul's 123
loss_list = []
for _dir in dirs:
    print(_dir)
    for mult in np.linspace(0, 1, 6):
        # for mult in [1.0]:
        dss = opf.DSSToCSVConverter(_dir, s_base=1e6, v_min=0, v_max=2)
        dss.dss.Solution.LoadMult(mult)
        dss.dss.Solution.Solve()
        dss.update()
        case = opf.create_case(
            _dir,
            n_steps=1,
            start_step=0,
            ignore_schedule=True,
            ignore_bat=True,
            ignore_gen=True,
        )
        case.modify(load_mult=mult)
        r: opf.PowerFlowResult = case.run_opf(
            wrapper="pyomo",
            control_regulators=False,
            control_capacitors=False,
            solver="highs",
        )
        v_df = r.voltages
        p_df = r.active_power_flows
        q_df = r.reactive_power_flows
        # Align both voltage DataFrames on bus name before comparing
        v_model = v_df.set_index("name").loc[:, ["a", "b", "c"]]
        v_dss = dss.v_solved.set_index("name").loc[:, ["a", "b", "c"]]
        common_names = v_model.index.intersection(v_dss.index)
        v_model = v_model.loc[common_names]
        v_dss = v_dss.loc[common_names]
        v_diff = (v_model - v_dss).abs()
        v_rdiff = v_diff / v_dss.abs()
        print(f"{mult:.2f}: V error %: {v_rdiff.max().max():.3%}")

