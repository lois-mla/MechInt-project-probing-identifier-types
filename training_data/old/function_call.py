def addition(x, y):
    return x + y

sum = FIM(2, 3)
>>>addition
#####
def hello_world():
    print("hello world")

FIM()
>>>hello_world
#####
def fahrenheit_to_celsius(fahrenheit):
  return (fahrenheit - 32) * 5 / 9

print(FIM(77))
>>>fahrenheit_to_celsius
#####
def my_function():
  pass
FIM()
>>>my_function
#####
def youngest(*kids):
  print("The youngest child is " + kids[2])

FIM("Emil", "Tobias", "Linus") 
>>>youngest
#####
def sum(*numbers):
  total = 0
  for num in numbers:
    total += num
  return total

print(FIM(1, 2, 3))
>>>sum
#####
def find_max(*numbers):
  if len(numbers) == 0:
    return None
  max_num = numbers[0]
  for num in numbers:
    if num > max_num:
      max_num = num
  return max_num

max_n = FIM(3, 7, 2, 9, 1)
>>> find_max
#####
def countdown(n):
  if n <= 0:
    print("Done!")
  else:
    print(n)
    countdown(n - 1)

FIM(5)
>>>countdown
#####
