import matplotlib.pyplot as plt
import fileinput

fig, ax = plt.subplots()

first_line = True

x_axis_title = "NO LINES"
x_axis = [0, 1]

for line in fileinput.input():
	cells = line.split(",") # Assume CSV string in English locale
	title = cells[0].strip()
	end = len(cells)
	# Ignore last cell if it's empty. That allows a trailing "," in
	# the CSV string
	if not cells[-1].strip():
		end = end - 1
	values = map(int, cells[1:end])
	if first_line:
		x_axis_title = title
		x_axis = values
		first_line = False
	else:
		line, = ax.plot(x_axis, values, '-', label=title)

ax.legend(loc='lower right')
plt.show()
