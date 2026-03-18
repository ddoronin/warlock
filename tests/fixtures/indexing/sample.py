def multiply(x: int, y: int) -> int:
    return x * y


class Counter:
    def __init__(self) -> None:
        self.value = 0

    def inc(self) -> None:
        self.value += 1
