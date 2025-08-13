def stepF(self, dirc):
    if(dirc>0):
        self.state.loc+=1
    else:
        self.state.loc-=1
    self.slider.set(self.state.loc)