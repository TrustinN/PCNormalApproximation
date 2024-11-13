from solver import PCSolver


class Core:
    def __init__(self, console, display):
        self.console = console
        self.display = display
        self.toggle_functions = {}
        self.solver = PCSolver()
        for keys in display.plotObjects.keys():
            self.toggle_functions[keys] = self.toggle_factory(keys)

        self.connect_pt()

    def connect_pt(self):
        self.params = self.console.options.params
        for key in self.toggle_functions:
            self.params.child("display").child(key).sigTreeStateChanged.connect(
                self.toggle_functions[key]
            )
        self.params.child("run").sigTreeStateChanged.connect(self.update)

    def update(self):
        # clear previous plots
        self.display.unloadItems()

        self.solver.setProp(self.params.child("Props").value())

        plotItems = self.solver.solve()
        self.display.loadItems(plotItems)
        self.params.child("display").child("mesh").setValue(True)
        self.params.child("display").child("tangent centers").setValue(True)

        profile = self.solver.profile
        self.console.updateData(profile)

    def toggle_factory(self, name):
        def toggle_func():
            on = self.params.child("display").child(name).value()
            if on:
                self.display.show(name)
            else:
                self.display.hide(name)

        return toggle_func
