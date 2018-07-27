from os import listdir
from os.path import isfile, join
import pandas as pd


def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

mypath = "/home/felix/.kaggle/datasets"
users = [f for f in listdir(mypath) if join(mypath, f)]

print len(users)

csv_list = []

projectnames = []

for user in users:
    mypath = "/home/felix/.kaggle/datasets/" + str(user)
    projects = [f for f in listdir(mypath) if join(mypath, f)]

    for project in projects:
        mypath = "/home/felix/.kaggle/datasets/" + str(user) + "/" + str(project)

        projectnames.append(str(user) + "_" + str(project))

        csv_files = find_csv_filenames(mypath)
        csv_list.extend(csv_files)


print "projects: " + str(len(projectnames))
print "csvs: " + str(len(csv_list))



