r = int(input("Enter number of rows: "))

for i in range(1, r + 1):
    space = r - i
    star = i
    print(" " * space, end="")
    print("* " * star)
