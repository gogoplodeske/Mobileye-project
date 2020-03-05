class Singleton:
   __instance = None
   @staticmethod
   def getInstance():
      """ Static access method. """
      if Singleton.__instance == None:
         Singleton()
      return Singleton.__instance
   def __init__(self):
      """ Virtually private constructor. """
      if Singleton.__instance != None:
         raise Exception("This class is a singleton!")
      else:
         Singleton.__instance = self


class TFL_Man(Singleton):
    prevframe={}
    currentframe = {}
    def __init__(self, frameID,framepath):
        super().__init__()
        self.currentframe['FramId'] = frameID
        self.currentframe['FramePath']=framepath
    def Setframe(self,frameID,framepath):
        self.prevframe=self.currentframe.copy()
        self.currentframe['FramId'] = frameID
        self.currentframe['FramePath'] = framepath
    

