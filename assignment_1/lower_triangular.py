r = int(input("Enter number of rows: "))

for i in range(1, r+ 1):
    for j in range(i):
        print("*", end=" ")
    print()
