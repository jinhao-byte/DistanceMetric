import arff
import copy
def Get_instValue(inst,att,info): # 获得 inst 的第att个属性的属性值
    Value_Norm = inst[att]
    return info[att][1].index(Value_Norm)
def Pre(Src):
    # 读取 ARFF 文件
    with open(Src, 'r') as file:
        data = arff.load(file)
    info_Data = data['attributes'] # 属性信息
    Data = data['data'] #所有数据
    #random.shuffle(Data) # 行次序随机打乱
    # 分成十折 方便后续进行十折交叉验证
    # 计算每份应该包含的行数
    total_rows = len(Data)
    num_parts = 10
    rows_per_part = total_rows // num_parts
    remainder = total_rows % num_parts
    # 初始化一个空列表来存储划分后的部分
    parts = []
    # 开始划分
    start_index = 0
    for i in range(num_parts):
        # 如果还有余数，给当前部分多分配一行
        end_index = start_index + rows_per_part + (1 if i < remainder else 0)
        parts.append(Data[start_index:end_index])
        start_index = end_index

    # parts,存储了10份均匀的数据样本集
    '''
    [   [[inst1],[inst3],[inst4],[inst5]],
        [[inst1],[inst3],[inst4],[inst5]],
        .....
        [[inst1],[inst3],[inst4],[inst5]],
    ]
    '''
    # info_Data,存储了数据集的属性信息
    '''
    [   ('buying', ['vhigh', 'high', 'med', 'low']), 
        ('maint', ['vhigh', 'high', 'med', 'low']), 
        ('doors', ['2', '3', '4', '5more']), 
        ('persons', ['2', '4', 'more']), 
        ('lug_boot', ['small', 'med', 'big']),
        ('safety', ['low', 'med', 'high']), 
        ('class', ['unacc', 'acc', 'good', 'vgood'])    ]
    '''
    return parts,info_Data

def split(parts,i):
    # 将第i部分的样本集合，当作测试集
    m_Test = parts[i-1]
    Copy_parts = copy.deepcopy(parts)
    del Copy_parts[i-1]
    m_Train = [sublist for inner_list in Copy_parts for sublist in inner_list]
    return m_Train,m_Test


def count1(Train, info):
    # 更新m_TotalAttValues , m_NumClasses , m_NumAttributes , m_NumInstances , m_ClassIndex
    # NumClasses
    NumClasses = int(len(info[-1][1]))
    # ClassIndex
    ClassIndex = int(len(info) - 1)
    # NumInstances
    NumInstances = len(Train)
    # NumAttributes
    NumAttributes = len(info)
    # TotalAttValues 数据集的所有属性值总数
    TotalAttValues = 0
    for i in range(0, len(info)-1):
        TotalAttValues = TotalAttValues + len(info[i][1])

    return TotalAttValues, NumClasses, NumAttributes, NumInstances, ClassIndex
def Get_Instances_Attribute_NumVlaues(info,i):
    # 数据集下，第i个属性的属性值总数
    '''
    [   ('buying', ['vhigh', 'high', 'med', 'low']),
        ('maint', ['vhigh', 'high', 'med', 'low']),
        ('doors', ['2', '3', '4', '5more']),
        ('persons', ['2', '4', 'more']),
        ('lug_boot', ['small', 'med', 'big']),
        ('safety', ['low', 'med', 'high']),
        ('class', ['unacc', 'acc', 'good', 'vgood'])    ]
    '''
    return len(info[i][1])
def Get_Instance_ClassValue(k,Train,info): # 得到第k个实例的类属性值
    inst = Train[k]
    Class_noum = inst[-1]
    # 找到 Class_noum 在 列表info[-1][1] 的索引
    ClassValue = info[-1][1].index(Class_noum)
    return ClassValue
def Get_Instance_AttributeValue(k,i,Train,info):# 得到第k个实例的第i个属性的属性值
    inst = Train[k]
    Att = inst[i]
    AttValue = info[i][1].index(Att)
    return AttValue

def Get_instClassIndex(classValue,info):
    #返回实例的类标签索引
    index = info[-1][1].index(classValue)
    return index


def count2(TotalAttValues, NumClasses, NumAttributes, NumInstances, ClassIndex,Train,info):
    # ClassAttCounts 初始化
    ClassAttCounts = [[0 for _ in range(TotalAttValues)] for _ in range(NumClasses)]
    # AttCounts 初始化
    AttCounts = [0 for _ in range(TotalAttValues)]
    # NumAttValues 初始化
    NumAttValues = [0 for _ in range(NumAttributes)]
    # StartAttIndex
    StartAttIndex = [0 for _ in range(NumAttributes)]

    # 进行数值填充
    # 更新 StartAttIndex NumAttValues
    TotalAttValues_TEMP = 0
    for i in range(0,NumAttributes):
        if i != ClassIndex:
            StartAttIndex[i] = TotalAttValues_TEMP
            NumAttValues[i] = Get_Instances_Attribute_NumVlaues(info,i)# 数据集下，第i个属性的属性值总数
            TotalAttValues_TEMP = TotalAttValues_TEMP + NumAttValues[i]
        else:
            StartAttIndex[i] = -1
            NumAttValues[i] = NumClasses
    # 更新 ClassAttCounts AttCounts
    for k in range(0,NumInstances):
        classVal = Get_Instance_ClassValue(k,Train,info)# 得到第k个实例的类属性值
        attIndex = [0]*NumAttributes
        for i in range(0,NumAttributes):
            if i == ClassIndex:
                attIndex[i] = -1
            else:
                attIndex[i] = StartAttIndex[i] + Get_Instance_AttributeValue(k,i,Train,info)
                AttCounts[attIndex[i]] = AttCounts[attIndex[i]] + 1
                ClassAttCounts[classVal][attIndex[i]] = ClassAttCounts[classVal][attIndex[i]] + 1
    return ClassAttCounts,AttCounts,NumAttValues,StartAttIndex