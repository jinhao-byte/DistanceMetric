import utils
from VDM import VDM
from ISCDM import ISCDM
src = 'data/car.arff'
parts,info_Data = utils.Pre(src)
m_Train,m_Test = utils.split(parts,1)
m_TotalAttValues , m_NumClasses , m_NumAttributes , m_NumInstances , m_ClassIndex = utils.count1(m_Train,info_Data)
m_ClassAttCounts , m_AttCounts , m_NumAttValues , m_StartAttIndex = utils.count2(m_TotalAttValues,m_NumClasses,m_NumAttributes,m_NumInstances,m_ClassIndex,m_Train,info_Data)
vdm = VDM(m_TotalAttValues , m_NumClasses , m_NumAttributes , m_NumInstances , m_ClassIndex,m_ClassAttCounts , m_AttCounts , m_NumAttValues , m_StartAttIndex,info_Data)
iscdm = ISCDM(m_TotalAttValues , m_NumClasses , m_NumAttributes , m_NumInstances , m_ClassIndex,m_ClassAttCounts , m_AttCounts , m_NumAttValues , m_StartAttIndex,info_Data)
#测试样本之间的距离
distance = vdm.Distance(m_Test[1],m_Test[2])
distance1 = iscdm.Distance(m_Test[1],m_Test[2])
distance3 = vdm.Distance(m_Test[1],m_Test[1])
distance4 = iscdm.Distance(m_Test[2],m_Test[2])
print(distance)
print(distance1)



