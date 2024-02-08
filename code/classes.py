class Individual:
    def __init__(self, code, amount_of_bins=0):
        self.code = code
        self.fitness = -1
        self.amount_of_bins = amount_of_bins

    def __repr__(self):
        if self.amount_of_bins == 0:
            return f"Individual(code={self.code}, fitness={self.fitness})"
        else:
            return f"Bins={self.amount_of_bins}, fitness={self.fitness}, individual(code={self.code}"


class Bin:
    def __init__(self, remaining_capacity):
        self.remaining_capacity = remaining_capacity
        self.items = []

    def add(self, weight):
        self.items.append(weight)
        self.remaining_capacity -= weight

    def __repr__(self):
            return f"Items={self.items}, Empty space={self.remaining_capacity}"
