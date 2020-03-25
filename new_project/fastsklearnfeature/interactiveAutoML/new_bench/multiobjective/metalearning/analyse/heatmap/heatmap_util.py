import numpy as np

def pivot2latex(my_pivot):
	values = my_pivot.values
	axes = my_pivot.axes

	my_latex = '[,xlabel=' + axes[1].name +',ylabel=' + axes[0].name + ', xtick={'
	for i in range(len(axes[1])):
		my_latex += str(i) + ', '
	my_latex = my_latex[:-2]
	my_latex += '} ,xticklabels={'
	for i in range(len(axes[1])):
		my_latex += str(axes[1][i]) + ', '
	my_latex = my_latex[:-2]
	my_latex += '} , ytick={'
	for i in range(len(axes[0])):
		my_latex += str(i) + ', '
	my_latex = my_latex[:-2]
	my_latex += '} ,yticklabels={'
	for i in range(len(axes[0])):
		my_latex += str(axes[0][len(axes[0]) - 1 - i]) + ', '
	my_latex = my_latex[:-2]
	my_latex += "}, x tick label style={rotate=90,anchor=east}, axis line style={draw=none}, x=1cm,y=1cm, x label style={at={(axis description cs:0.5,-0.05)},anchor=north}]\n\\addplot+[sharp plot, opacity=0.0] coordinates {(0,0)("+ str(len(axes[1])-1) +","+ str(len(axes[0])-1) +")};\n"


	for y_value_i in range(values.shape[0]):
		for x_value_i in range(values.shape[1]):

			if not np.isnan(values[y_value_i, x_value_i]):
				start_point_x = -0.5 + x_value_i
				start_point_y = len(axes[0]) - y_value_i - 1.5
				my_latex +="\\fill [color" + str(int(values[y_value_i, x_value_i])) + "] ("+ str(start_point_x) +"cm," + str(start_point_y) + "cm) rectangle ("+ str(start_point_x+1) +"cm," + str(start_point_y+1) + "cm);\n"

	print(my_latex)