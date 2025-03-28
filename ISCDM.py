import math

import utils
class ISCDM:
    def __init__(self,m_TotalAttValues , m_NumClasses , m_NumAttributes , m_NumInstances , m_ClassIndex,m_ClassAttCounts , m_AttCounts , m_NumAttValues , m_StartAttIndex,info):
        self.m_TotalAttValues = m_TotalAttValues
        self.m_NumClasses = m_NumClasses
        self.m_NumAttributes = m_NumAttributes
        self.m_NumInstances = m_NumInstances
        self.m_ClassIndex = m_ClassIndex
        self.m_ClassAttCounts = m_ClassAttCounts
        self.m_AttCounts = m_AttCounts
        self.m_NumAttValues = m_NumAttValues
        self.m_StartAttIndex = m_StartAttIndex
        self.info = info
        self.ClassCounts = self.ClassCounts(self.m_ClassAttCounts,self.m_NumClasses)
    def ClassCounts(self,m_ClassAttCounts,m_NumClasses):
        count = [0]*self.m_NumClasses
        for i in range(0,m_NumClasses):
            for k in range(0,self.m_StartAttIndex[1]):
                count[i] = count[i] + m_ClassAttCounts[i][k]
        return count
    # 计算 first 和 second 的距离
    def Distance(self,first,second):
        attIndexf = [0]*self.m_NumAttributes
        for att in range(0,self.m_NumAttributes):
            if att == self.m_ClassIndex:
                attIndexf[att] = -1
            else:
                attIndexf[att] = self.m_StartAttIndex[att] + utils.Get_instValue(first,att,self.info)
        attIndexs = [0]*self.m_NumAttributes
        for att in range(0,self.m_NumAttributes):
                if att == self.m_ClassIndex:
                    attIndexs[att] = -1
                else:
                    attIndexs[att] = self.m_StartAttIndex[att] + utils.Get_instValue(second,att,self.info)
        classVal_train = utils.Get_instValue(second,self.m_ClassIndex,self.info)        #得到实例的类标签值
        distance = 0.0
        for att in range(0,self.m_NumAttributes):
            if attIndexf[att] == -1 or attIndexf[att] == -1 :
                continue
            elif utils.Get_instValue(first,att,self.info) == utils.Get_instValue(second,att,self.info):
                distance = distance + 0
            else:# distance+=Math.pow((1-(m_ClassAttCounts[classVal_train][attIndexf[att]])/(m_ClassCounts[classVal_train])), 2);
                distance = distance +(1-(self.m_ClassAttCounts[classVal_train][attIndexf[att]])/(self.ClassCounts[classVal_train]))**2
        distance = math.sqrt(distance)
        return distance





