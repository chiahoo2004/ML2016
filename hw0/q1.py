from sys import argv
import numpy

data = numpy.loadtxt(argv[2])
result = sorted( data[ : , argv[1] ] )

o_file = open('ans1.txt', 'w')
for i in result[:-1]:
	o_file.write(str(i)+',')
o_file.write(str(result[-1]))
