
def testFoo(a_):
    b = 2
    def innerFoo1():
        print(a_)
        return 12
    def innerFoo2():
        y = innerFoo1()
        return y
    x = innerFoo2()
    return x

a = 1
testFoo(a)

