#scalar
age = 19
print(age)


#vector
age = 19
height = 161
weight = 60
vector = [age,height,weight]
print(vector)


#matrix
age = ([10,20],[30,40])
for row in age:
  print(row)


#AND gate
a = int(input("Enter the value of a as either 0 or 1: "))
b = int(input("Enter the value of B as either 0 or 1: "))
print(a and b)


#OR gate
a = int(input("Enter the value of a as either 0 or 1: "))
b = int(input("Enter the value of B as either 0 or 1: "))
print(a or b)


#XOR gate
a = int(input("Enter the value of a as either 0 or 1: "))
b = int(input("Enter the value of B as either 0 or 1: "))
print(a ^ b)



#numpy
import numpy as np
x = np.array([[10,20],[30,40]])
print(x)
