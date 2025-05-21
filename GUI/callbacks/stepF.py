def stepF(self, dirc):
    if(dirc>0):
        self.loc+=1
    else:
        self.loc-=1
    self.slider.set(self.loc)