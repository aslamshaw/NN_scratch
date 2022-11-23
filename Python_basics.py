"""Basics of Python"""
# Using libraries
from math import *
print(sqrt(25))
Phrase = "You are already dead!"
print(Phrase.upper().islower())
print(Phrase.index("r"))
print(Phrase.replace("You", "Omai wa"))
my_num = Phrase.index("r")
print(str(Phrase.index("r")) + ". " + Phrase)
print(pow(Phrase.index("r"), 2), max(Phrase.index("r"), 6))
Name = input("Enter your name: ")
print("Hello " + Name)


# Basic Calculator
n1 = input("Enter your 1st Number ")
n2 = input("Enter your 2nd Number ")
c = input("What do you want to do?\nAdd\nSub\nMul\nDiv\n")
if c == "Add":
    print(float(n1) + float(n2))
elif c == "Sub":
    print(float(n1) - float(n2))
elif c == "Mul":
    print(float(n1) * float(n2))
elif c == "Div":
    print(float(n1) / float(n2))


# Array indexing
Array = ["index 0", "index 1", "index 2", "index 3", "index 4"]
print(Array[0], Array[-5])
print(Array[0:4]) #Index 4 is excluded
new_indices = ["index 5", "index 6"]
Array.extend(new_indices)
Array.append("index 7")
print(Array.index("index 7"))

for index in range(len(Array)):
    print(index)
    print(Array[index])

grid = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
for i in grid:
    for j in i:
        print(j)
print(grid[2][0])


# Using functions
def say_hi(name, age):
    x = "Hi " + name + ", you are " + age + " years old."
    return x
print(say_hi(input("Type your name: "), input(str("Type your age: "))))
print(say_hi("Bob", "2000"))

def raise_to_power(num, power):
    result = 1
    for i in range(power):
        result = result * num
    return result
print(raise_to_power(4, 2))


# Name guessing game
secret = "Aslam"
guess = ""
count = 0
limit = 3
out_of_guesses = False
while guess != secret and out_of_guesses == False:
    if count < limit:
        guess = input("Guess the name")
        count += 1
    else:
        out_of_guesses = True
if out_of_guesses:
    print("U lose")
else:
    print("U win")

# Phrase changing game
def translate(phrase):
    translation = ""
    for letter in phrase:
        if letter.lower() in "aeiou":
            if letter.isupper():
                translation = translation + "G"
            else:
                translation = translation + "g"
        else:
            translation = translation + letter
    return translation
print(translate(input("What to you want to translate?\n")))


# Odd even using logic
def odd_even_AND(num):
    result = "Number is even"
    if num & 1:
    # if num & 1 == 1:
        result = "Number is odd"
    return result

l1 = ["Bhindi", "Aloo", "chopsticks", "chowmein"]
# i = 1
# for item in l1:
#     if i%2:
#         print(f"Jarvis please buy {item}")
#     i += 1
for index, item in enumerate(l1):
    if index%2 == 0:
        print(f"Jarvis please buy {item}")

