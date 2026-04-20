import pandas as pd
df = pd.read_csv(r"D:\New folder (2)\NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv", nrows=5)
cols = df.columns.tolist()
_leak_tags = ("fev1_fvc","fev1fvc","ratio_z","ratio5th","ratio2point5","obstruction_","normal_","mixed_","prism_","restrictive_","onevariable_lln")
leak = [c for c in cols if c != "Obstruction" and any(t in c.lower() for t in _leak_tags)]
if "Baseline_FEV1_FVC_Ratio" in cols:
    leak.append("Baseline_FEV1_FVC_Ratio")
leak = sorted(set(leak))
keep = [c for c in cols if c not in leak]
print(f"Total cols: {len(cols)}")
print(f"Leakage dropped: {len(leak)}")
print(f"Remaining (incl target+id): {len(keep)}")
print("\n--- DROPPED ---")
for c in leak:
    print(f"  {c}")
print("\n--- KEPT ---")
for c in keep:
    print(f"  {c}")
