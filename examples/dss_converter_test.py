from distopf.api import create_case
from distopf import CASES_DIR

case = create_case(CASES_DIR / "dss" / "ieee13_dss/IEEE13Nodeckt.dss")
print(case.branch_data)
print()
