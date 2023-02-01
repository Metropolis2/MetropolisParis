import os, sys
import subprocess


# creating a new run folder of number n+1 to store the newest run
outpath = "run"+str(max((int(f[3:]) for f in os.listdir() if f.startswith("run")), default=0)+1)
os.mkdir(outpath)

# set input data path
datapath= f"./output/strasbourg_run"

# script path (leave as "" if your command is in the current folder)
scriptpath = r"C:\Users\theot\OneDrive\TheseMaster\Models\Metropolis\Rust\0.1.7/"

#add ".exe" if on windows
syst = ""
if sys.platform == "win32":
    syst=".exe"

command = f"{scriptpath}metropolis{syst} --agents {datapath}/agents.json  --parameters {datapath}/parameters.json --road-network {datapath}/network.json  --output {outpath}"
subprocess.run(command, shell=True)
print(f"Saved in {outpath}")
