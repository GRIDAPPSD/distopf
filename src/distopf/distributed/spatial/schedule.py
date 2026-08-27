import numpy as np
import pandas as pd


def add_v_swing_to_schedules(
    schedules: pd.DataFrame,
    v: pd.DataFrame,
    receiving_area: str,
) -> pd.DataFrame:
    v_rows = v.loc[v.name.astype(str) == str(receiving_area), ["t", "a", "b", "c"]]
    assert not v_rows.duplicated(["t"]).any(), (
        f"Duplicate voltage boundary values for area {receiving_area}"
    )
    v_swing = v_rows.rename(
        columns={"t": "time", "a": "v_a", "b": "v_b", "c": "v_c"}
    ).set_index("time")
    schedules = schedules.set_index("time")
    assert v_swing.index.isin(schedules.index).all(), (
        f"Voltage boundary times are missing from the schedule "
        f"for area {receiving_area}"
    )
    schedules.loc[v_swing.index, ["v_a", "v_b", "v_c"]] = v_swing[["v_a", "v_b", "v_c"]]
    return schedules.reset_index()


def add_v_down_to_schedules(
    schedules: pd.DataFrame,
    v: pd.DataFrame,
    sending_area: str,
) -> pd.DataFrame:
    v_rows = v.loc[:, ["t", "a", "b", "c"]]
    assert not v_rows.duplicated(["t"]).any(), (
        f"Duplicate voltage boundary values for area {sending_area}"
    )
    v_prep = v_rows.rename(columns={"t": "time"}).set_index("time")
    schedules = schedules.set_index("time")
    assert v_prep.index.isin(schedules.index).all(), (
        f"Voltage boundary times are missing from the schedule for area {sending_area}"
    )
    v_cols = [f"{sending_area}.{phase}.v" for phase in "abc"]
    for col in v_cols:
        if col not in schedules.columns:
            schedules[col] = np.nan
    v_data = v_prep[["a", "b", "c"]].copy()
    v_data.columns = v_cols
    schedules.loc[v_prep.index, v_cols] = v_data
    return schedules.reset_index()


def add_s_to_schedules(
    schedules: pd.DataFrame,
    s: pd.DataFrame,
    sending_area: str,
) -> pd.DataFrame:
    p_cols = [f"{sending_area}.{phase}.p" for phase in "abc"]
    q_cols = [f"{sending_area}.{phase}.q" for phase in "abc"]
    s_rows = s.loc[:, ["t", "a", "b", "c"]]
    assert not s_rows.duplicated(["t"]).any(), (
        f"Duplicate power boundary values for area {sending_area}"
    )
    s_prep = s_rows.rename(columns={"t": "time"}).set_index("time")
    schedules = schedules.set_index("time")
    assert s_prep.index.isin(schedules.index).all(), (
        f"Power boundary times are missing from the schedule for area {sending_area}"
    )
    p_data = s_prep[["a", "b", "c"]].apply(np.real)
    p_data.columns = p_cols
    q_data = s_prep[["a", "b", "c"]].apply(np.imag)
    q_data.columns = q_cols
    schedules.loc[s_prep.index, p_cols] = p_data
    schedules.loc[s_prep.index, q_cols] = q_data
    return schedules.reset_index()
