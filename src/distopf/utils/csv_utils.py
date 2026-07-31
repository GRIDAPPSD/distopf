

def branch_data_csv_upgrade(branch_data, bus_data):
    """
    - add columns from_name and to_name
    - reorder columns for readability
    
    Example
    > import pandas as pd
    > from pathlib import Path
    > base = Path(__file__).parent
    > branch = pd.read_csv(base/"branch_data.csv")
    > bus = pd.read_csv(base/"bus_data.csv")
    > brach = branch_data_csv_upgrade(branch, bus)
    > branch.to_csv(base/"branch_data_cleaned.csv", index=False)
    """
    id_map = dict(zip(bus["id"], bus["name"]))
    branch["from_name"] = branch["fb"].map(id_map)
    branch["to_name"] = branch["tb"].map(id_map)
    first = ["fb", "tb", "from_name", "to_name", "name", "type", "status", "phases"]
    branch = branch[first + [c for c in branch.columns if c not in first]]
    return branch
