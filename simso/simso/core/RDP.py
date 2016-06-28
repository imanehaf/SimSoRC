# coding=utf-8


class RDP(object):
    def __init__(self, mt):
        self.prd = self.get_reuse(mt)
    
    
    def get_reuse(self, mt):
        """
        Get reuse distance for a given memory trace
        """
        # Reuse distance
        self.rd = {}
        prevmt = []
        prevNum = []
    
        for i in range(len(mt)):
            w = mt[i]
            if w in prevmt:
                K = i-prevNum[prevmt.index(w)]-1
                if K in self.rd.keys():
                    self.rd[K] = self.rd[K]+1
                else:
                    self.rd[K] = 1
                prevNum[prevmt.index(w)] = i
            else: # first time use
                prevmt.append(w)
                prevNum.append(i)
                if 'inf' in self.rd.keys():
                    self.rd['inf'] = self.rd['inf']+1
                else:
                    self.rd['inf'] = 1
        return self.rd

