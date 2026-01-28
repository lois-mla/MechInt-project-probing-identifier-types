def FIM(x, y):
    return x + y

sum = addition(2, 3)
>>>addition
#####
def FIM():
    print("hello world")

hello_world()
>>>hello_world
#####
def FIM(fahrenheit):
  return (fahrenheit - 32) * 5 / 9

print(fahrenheit_to_celsius(77))
>>>fahrenheit_to_celsius
#####
def FIM():
  pass
my_function()
>>>my_function
#####
def FIM(*kids):
  print("The youngest child is " + kids[2])

youngest("Emil", "Tobias", "Linus") 
>>>youngest
#####
def FIM(*numbers):
  total = 0
  for num in numbers:
    total += num
  return total

print(sum(1, 2, 3))
>>>sum
#####
def FIM(*numbers):
  if len(numbers) == 0:
    return None
  max_num = numbers[0]
  for num in numbers:
    if num > max_num:
      max_num = num
  return max_num

max_n = find_max(3, 7, 2, 9, 1)
>>> find_max
#####
def FIM(n):
  if n <= 0:
    print("Done!")
  else:
    print(n)
    countdown(n - 1)
>>>countdown
#####
def countdown(n):
  if n <= 0:
    print("Done!")
  else:
    print(n)
    FIM(n - 1)
>>>countdown
#####
def FIM(n):
  if n <= 0:
    print("Done!")
  else:
    print(n)
    countdown(n - 1)

countdown(5)
>>>countdown
#####
def countdown(n):
  if n <= 0:
    print("Done!")
  else:
    print(n)
    FIM(n - 1)

countdown(5)
>>>countdown
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
FIM = 0
var += 1
#####
class FIM:
    def __init__(self):
        self.data = []

    def add(self, x):
        self.data.append(x)

bag = Bag()
#####

