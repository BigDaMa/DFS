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

#users = ['dansbecker']

schema_file = open('/tmp/schema.csv', 'w+')

for user in users:
    mypath = "/home/felix/.kaggle/datasets/" + str(user)
    projects = [f for f in listdir(mypath) if join(mypath, f)]

    for project in projects:
        mypath = "/home/felix/.kaggle/datasets/" + str(user) + "/" + str(project)
        csv_files = find_csv_filenames(mypath)
        csv_list.extend(csv_files)

        print csv_files

        for csv_file in csv_files:
            mypath = "/home/felix/.kaggle/datasets/" + str(user) + "/" + str(project) + "/" + csv_file
            pandas_table = pd.read_csv(mypath)
            print "Project: " + str(project)
            print pandas_table.head(n=4)

            print pandas_table.columns


            usable = raw_input(str(mypath) + " Is it usable? (T)rain, (E)valuation, (N)o")

            if usable == 'T' or usable == 'E':
                tasktype = raw_input(str(user) + ": " + str(project) + " is (R)egression, (C)lassification, or (N)one?")

                header_lines = raw_input("Choose number of header lines?")


                for column_i in range(len(pandas_table.columns)):
                    print str(column_i) + " <" + pandas_table.columns[column_i] + ">"

                target_column = raw_input("Choose column number?")

                schema_file.write(str(user) + "#" + str(project) + "#" + csv_file + "#" + str(usable) + "#" + str(tasktype) + "#" + str(target_column) + "#" + str(header_lines) + "\n")
                schema_file.flush()
schema_file.close()

