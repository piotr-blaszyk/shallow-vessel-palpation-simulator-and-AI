class Bar:
    def __init__(self):
        self.foo = 1
    
    def a(self):
        self.foo = 2
    
    def b(self):
        print(self.foo)

bar = Bar()
bar.b()
bar.a()
bar.b()
