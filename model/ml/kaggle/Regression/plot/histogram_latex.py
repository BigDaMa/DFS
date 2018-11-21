
def plot(fscore_list, single_col_fscore_list, datasets):

    latex = """
    \\documentclass{article}
    \\usepackage[utf8]{inputenc}
    \\usepackage{filecontents}
    \\usepackage{pgfplots, pgfplotstable}
    \\usepgfplotslibrary{statistics}
    
    \\title{histogram2}
    \\author{Felix Neutatz}
    \\date{October 2018}
    """

    for data_i in range(len(fscore_list)):
        latex += """
        \\begin{filecontents}{data""" + str(data_i)  + """.csv}
        dist
        """
        for element_i in fscore_list[data_i]:
            latex += str(element_i) + "\n"

        latex += """
        \\end{filecontents}
        """

    latex += """
    \\begin{document}
    \\maketitle
    """

    for data_i in range(len(fscore_list)):

        datasetname = datasets[data_i][0].split("/")[-1].split('.')[0].replace('_', ' ')

        latex +="""
        \\begin{figure}[ht!]
            \\centering	
            \\begin{tikzpicture}
            \\begin{axis}[
                ybar,
                xmin=0,
                xmax=1,
                ymin=0
            ]
            \\addplot +[
                hist={
                    bins=50
                }   
            ] table [y index=0] {data""" + str(data_i) + """.csv};
            \\draw[line width=0.5mm,red] (rel axis cs: """ + str(single_col_fscore_list[data_i]) + """,0) -- (rel axis cs: """ + str(single_col_fscore_list[data_i]) + """, 1.0);
            \\end{axis}
            \\end{tikzpicture}
            \\caption{"""+ datasetname + """}
        \\end{figure}
        """

    latex += '\\end{document}'

    return latex