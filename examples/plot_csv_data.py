from __future__ import unicode_literals
import matplotlib.pyplot as plt
import fileinput
import sys

#
# Displays one ore more CSV files in a graph. Intended to be used
# with the `bench_tables.rs` example.
#
# Accepts data from STDIN and additional files can be passed in as
# command line arguments. A use case would be to display the current
# benchmark results in STDIN and a reference benchmark as a file.
#

def process_file(ax, filename, fileinput):
	first_line = True
	if filename == "STDIN":
		linestyle = "-"
	else:
		linestyle = "--"

	x_axis = [0, 1]
	for line in fileinput:
		cells = line.strip().split(",") # Assume CSV string in English locale
		title = filename + " " + cells[0].strip()
		end = len(cells)
		# Ignore last cell if it's empty. That allows a trailing "," in
		# the CSV string
		if not cells[-1].strip():
			end = end - 1
		values = map(int, cells[1:end])
		if first_line:
			x_axis = values
			first_line = False
		else:
			line, = ax.plot(x_axis, values, linestyle, label=title)

fig, ax = plt.subplots()
process_file(ax, "STDIN", fileinput.input("-", openhook=fileinput.hook_encoded("utf16")))
for filename in sys.argv[1:]:
	with open(filename, "r") as filehandle:
		process_file(ax, filename, fileinput.input(filename, openhook=fileinput.hook_encoded("utf16")))

ax.legend(loc='lower right')
plt.show()
