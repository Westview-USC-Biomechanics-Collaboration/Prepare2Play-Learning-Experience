class GlobalVariable:
    def __init__(self):
        self.sex = None # Default value, can be 'm' or 'f'
    def setMaleSex(self):
        self.sex = 'm'
        print("COM SEX SET TO MALE")

    def setFemaleSex(self):
        self.sex = 'f'
        print("COM SEX SET TO FEMALE")

globalVariable = GlobalVariable()
