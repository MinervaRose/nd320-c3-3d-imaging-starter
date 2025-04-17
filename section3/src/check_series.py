import os
import pydicom

root = "routed/final_study/study1"

for f in os.listdir(root):
    if f.endswith(".dcm"):
        dcm = pydicom.dcmread(os.path.join(root, f))
        print("SeriesDescription:", dcm.SeriesDescription)
        break
