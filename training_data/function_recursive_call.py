def countdown(n):
  if n <= 0:
    print("Done!")
  else:
    print(n)
    FIM(n - 1)
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
