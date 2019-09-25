import subprocess
import warnings
warnings.filterwarnings("ignore")

command = '/home/felix/anaconda/envs/new_project/bin/python /home/felix/FastFeatures/new_project/fastsklearnfeature/interactiveAutoML/CreditWrapper.py --features 1 0 1'

proc = subprocess.Popen(command,
                       shell=True,
                       stdout=subprocess.PIPE,
                       )

output = 0.0
while proc.poll() is None:
	output = float(proc.stdout.readline())
	break

print(output)