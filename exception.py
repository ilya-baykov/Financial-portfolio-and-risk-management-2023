class BadValue(Exception):
    def __init__(self, message="Пользовательское исключение"):
        self.message = "Неверное значение"
        super().__init__(self.message)
