import utils
class VDM:
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
        distance = 0.0
        for classVal in range(0,self.m_NumClasses):
            for att in range(0,self.m_NumAttributes):
                if attIndexf[att] == -1 or attIndexs[att] == -1:
                    continue
                else:
                    distance = distance + abs(((self.m_ClassAttCounts[classVal][attIndexf[att]]+1.0)/(self.m_AttCounts[attIndexf[att]]+self.m_NumClasses)
                                               -(self.m_ClassAttCounts[classVal][attIndexs[att]]+1.0)/(self.m_AttCounts[attIndexs[att]]+self.m_NumClasses)))
        return distance

#
# src = 'data/car.arff'
# parts,info_Data = utils.Pre(src)
# # print(parts)
# # print(info_Data)
# # 将parts，转变成训练集和测试集,split中的参数1，表示第一折
# m_Train,m_Test = utils.split(parts,1)
# m_TotalAttValues = 0
# m_NumClasses = 0
# m_NumAttributes = 0
# m_NumInstances = 0
# m_ClassIndex = -1
# m_TotalAttValues , m_NumClasses , m_NumAttributes , m_NumInstances , m_ClassIndex = utils.count1(m_Train,info_Data)
# m_ClassAttCounts , m_AttCounts , m_NumAttValues , m_StartAttIndex = utils.count2(m_TotalAttValues,m_NumClasses
#                                                                                  ,m_NumAttributes,m_NumInstances
#                                                                                  ,m_ClassIndex,m_Train,info_Data)
# # print(m_ClassAttCounts)
# # print(m_AttCounts)
# # print(m_NumAttValues)
# # print(m_StartAttIndex)


