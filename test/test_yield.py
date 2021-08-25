import numpy as np

def fun():
    yield 1
    print("to 2")
    yield 2

    yield 3

    print(3)


if __name__=="__main__":
    asd = fun()
    for i in asd:
        print(i)