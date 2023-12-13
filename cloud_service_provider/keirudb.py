from enum import Enum
from threading import Timer, Condition

class State(Enum):
    WAITING = 1
    FINISHED = 2

class Keiru:
    def __init__(self, timer=60):
        self.destroy_time = timer
        self.database = {}
        
    def cleardb(self):
        self.database = {}

    def destroy(self, userid):
        if userid in self.database:
            del self.database[userid]
    
    def add(self, userid, newdb):
        self.finishpush(userid, newdb)
    
    def startpush(self, userid):
        self.database[userid] = {'db': [], 'state': State.WAITING, 'cv': Condition() }
    
    def finishpush(self, userid, newdb):
        self.database[userid]['db'] = newdb
        self.database[userid]['state'] = State.FINISHED
        if self.database[userid]['cv'] is not None:
            with self.database[userid]['cv']:
                self.database[userid]['cv'].notify_all()
        timer = Timer(self.destroy_time, self.destroy, args=[userid])
        timer.start()
    
    def getdata(self, userid):
        if userid not in self.database:
            return None
        
        while self.database[userid]['state'] == State.WAITING:
            with self.database[userid]['cv']:
                self.database[userid]['cv'].wait()
        
        return self.database[userid]['db']
        