import numpy as np
from operator import itemgetter
from collections import deque

class MicroUnits:
    def __init__(self, Broodwar):
        self.units = {}
        self.controlledUnits = {}
        self.memory = deque([])
        self.mem_len = 10000000
        self.Broodwar = Broodwar

    def addUnit(self, unit, controlled=False):
        if controlled:
            self.controlledUnits[unit.getID()] = (unit, unit.getHitPoints()+unit.getShields())
        self.units[unit.getID()] = (unit, unit.getHitPoints()+unit.getShields())

    def removeUnit(self, unit, controlled=False):
        if controlled:
            self.controlledUnits.pop(unit.getID(), None)
        self.units.pop(unit.getID(), None)

    def lostHealth(self):
        ret = 0
        for k, v in self.units.items():
            ret += v[1] - v[0].getHitPoints()+v[0].getShields()
        return ret

    def act(self, game_map):
        for k, v in self.controlledUnits.items():
            '''Placeholder - Model will make predictions here'''
            return
        
    def _addMem(self, mem):
        if len(self.memory) > self.mem_len:
            self.memory.popleft()
        self.memory.append(mem)   
        
    def attack(self, unit, val):
        units = Broodwar.enemy().getUnits()
        opts = []
        lowest_hp_in_range = (None, -1)
        highest_dps_in_range(None, -1)
        lowest_hp = (None, -1)
        highest_dps(None, -1)
        for e in units:
            weapon = e.getType().groundWeapon()
            if unit.isInWeaponRange(e):
                if e.getHitPoints()+e.getShields() < lowest_hp_in_range[1]:
                    lowest_hp_in_range = (e, e.getHitPoints()+e.getShields())
                elif weapon.DamageAmount()*weapon.damageFactor() > highest_dps_in_range[1]:
                    highest_dps_in_range = (e, weapon.DamageAmount()*weapon.damageFactor())
                opts.append((e,unit.getDistance(e)))
        opts.sort(key=itemgetter(1))
        if val < 3:
            try:
                u = opts[val][0]
                if u != None:
                    unit.attack(u)
            except IndexError:
                return
        elif val == 4 and lowest_hp_in_range[0] != None:
            unit.attack(lowest_hp_in_range[0])
        elif val == 5 and highest_dps_in_range[0] != None:
            unit.attack(highest_dps_in_range[0])
        elif val == 6 and lowest_hp[0] != None:
            unit.attack(lowest_hp[0])
        elif val == 7 and highest_dps[0] != None:
            unit.attack(highest_dps[0])
        return
    
    def move(self, unit, val):
        top = unit.getTop()
        bottom = unit.getBottom()
        right = unit.getRight()
        left = unit.getLeft()
        x = right - (right-left)/2
        y = bottom - (bottom-top)/2
        if val == 0:
            return #no move
        elif val == 1:
            unit.move(cybw.Position(x=x, y=y-32)) #up
        elif val == 2:
            unit.move(cybw.Position(x=x+32, y=y-32)) #up-right
        elif val == 3:
            unit.move(cybw.Position(x=x+32, y=y)) #right
        elif val == 4:
            unit.move(cybw.Position(x=x+32, y=y+32)) #down-right
        elif val == 5:
            unit.move(cybw.Position(x=x, y=y+32)) #down
        elif val == 6:
            unit.move(cybw.Position(x=x-32, y=y+32)) #down-left
        elif val == 7:
            unit.move(cybw.Position(x=x-32, y=y)) #left
        elif val == 8:
            unit.move(cybw.Position(x=x-32, y=y-32)) #up-left
        return

    def command(self, unit, val):
        if val < 9:
            move(unit, val)
        else:
            attack(unit, val-8)
        return
    
    def buildGameMap():
        game_map = np.zeros((256,256,16))
        for x in range(256):
            for y in range(256):
                game_map[x,y,0] = self.Broodwar.getGroundHeight(x=int(x/2), y=int(y/2))
                
        units = self.Broodwar.getAllUnits()
        for unit in units:
            if unit.exists() != True:
                continue
            info = getUnitInfo(unit)
            top, bottom, right, left = getUnitPosition(unit)
            for i in range(left, right+1):
                for j in range(top, bottom+1):
                    game_map[i,j,1:] = info
        return game_map

    def getUnitPosition(unit):
        top = int(round(unit.getTop()/16))
        bottom = int(round(unit.getBottom()/16))
        right = int(round(unit.getRight()/16))
        left = int(round(unit.getLeft()/16))
        return top, bottom, right, left
        
    def getUnitInfo(unit, pos=False):
        data = []
        data.append(unit.getHitPoints())
        data.append(unit.getShields())
        data.append(unit.sightRange())
        if unit.isFlyer():
            data.append(1)
        else:
            data.append(0)
        unitType = unit.getType()
        data.extend(getWeaponInfo(unitType.groundWeapon()))
        data.extend(getWeaponInfo(unitType.airWeapon()))
        sizeType = str(unitType.size())
        if sizeType == 'Small':
            data.extend([1,0,0])
        elif sizeType == 'Medium':
            data.extend([0,1,0])
        elif sizeType == 'Large':
            data.extend([0,0,1])
        else:
            data.extend([0,0,0])
        if pos:
            data.extend(getUnitPosition(unit))
        return np.asarray(data, dtype=np.float32)

    def getWeaponInfo(weapon):
        data = []
        data.append(weapon.damageAmount())
        data.append(weapon.damageBonus())
        data.append(weapon.maxRange())
        data.append(weapon.medianSplashRadius())
        data.append(weapon.damageFactor())
        return data
