import arff
import copy
global TotalAttValues,NumClasses,NumAttributes,NumInstances,ClassIndex,ClassAttCounts,AttCounts,NumAttValues,StartAttIndex,info
TotalAttValues  = None
NumClasses = None
NumAttributes = None
NumInstances = None
ClassIndex = None
ClassAttCounts = [[]]
AttCounts = []
NumAttValues = None
StartAttIndex = []
info = None
# 指定超参数K
K = 10
# 交叉验证
folds = 10
# 10折交叉验证次数
Number_of_repetitions = 10
# 指定数据集路径
src = 'data/flags.arff'
def split(parts,i):
    # 将第i部分的样本集合，当作测试集
    m_Test = parts[i-1]
    Copy_parts = copy.deepcopy(parts)
    del Copy_parts[i-1]
    m_Train = [sublist for inner_list in Copy_parts for sublist in inner_list]
    return m_Train,m_Test
def Pre(Src):
    # 读取 ARFF 文件
    with open(Src, 'r') as file:
        data = arff.load(file)
    info = data['attributes'] # 属性信息
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
    return parts,info
class NeighInstances:
    def __init__(self, K):
        self.K = K
        self.instances = []

    def add_instance(self, label, distance):
        # 如果实例数量没有达到K，则直接添加
        if len(self.instances) < self.K:
            self.instances.append((label, distance))
            self._sort_instances()
        else:
            # 如果达到K个实例，检查当前距离是否小于最远的邻居的距离
            max_distance = self.instances[-1][1]
            if distance < max_distance:
                # 替换最远的邻居
                self.instances[-1] = (label, distance)
                self._sort_instances()

    def _sort_instances(self):
        # 按照距离对实例进行排序
        self.instances.sort(key=lambda x: x[1])

    def get_instances(self):
        # 返回当前的邻居实例集合
        return self.instances
def Get_instValue(inst,att,info): # 获得 inst 的第att个属性的属性值
    Value_Norm = inst[att]
    return info[att][1].index(Value_Norm)
def Get_instClassIndex(classValue,info):
    #返回实例的类标签索引
    index = info[-1][1].index(classValue)
    return index
def Get_Instance_AttributeValue(k,i,Train,info):# 得到第k个实例的第i个属性的属性值
    inst = Train[k]
    Att = inst[i]
    AttValue = info[i][1].index(Att)
    return AttValue
def Get_Instance_ClassValue(k,Train,info): # 得到第k个实例的类属性值
    inst = Train[k]
    Class_noum = inst[-1]
    # 找到 Class_noum 在 列表info[-1][1] 的索引
    ClassValue = info[-1][1].index(Class_noum)
    return ClassValue
def Get_Instances_Attribute_NumVlaues(info,i):
    # 数据集下，第i个属性的属性值总数
    return len(info[i][1])
def count(Train):
    global TotalAttValues, NumClasses, NumAttributes, NumInstances, ClassIndex, ClassAttCounts, AttCounts, NumAttValues, StartAttIndex,info
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
    for i in range(0, len(info) - 1):
        TotalAttValues = TotalAttValues + len(info[i][1])
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

def Distance(first,second):
    attIndexf = [0]*NumAttributes
    for att in range(0,NumAttributes):
        if att == ClassIndex:
            attIndexf[att] = -1
        else:
            attIndexf[att] = StartAttIndex[att] + Get_instValue(first,att,info)
    attIndexs = [0]*NumAttributes
    for att in range(0,NumAttributes):
            if att == ClassIndex:
                attIndexs[att] = -1
            else:
                attIndexs[att] = StartAttIndex[att] + Get_instValue(second,att,info)
    distance = 0.0
    for classVal in range(0,NumClasses):
        for att in range(0,NumAttributes):
            if attIndexf[att] == -1 or attIndexs[att] == -1:
                continue
            else:
                distance = distance + abs(((ClassAttCounts[classVal][attIndexf[att]]+1.0)/(AttCounts[attIndexf[att]]+NumClasses)
                                           -(ClassAttCounts[classVal][attIndexs[att]]+1.0)/(AttCounts[attIndexs[att]]+NumClasses)))
    return distance

def main():
    # 执行KNN算法
    global info
    parts, info = Pre(src)
    predict_List = []
    for fold in range(1, folds + 1):
        m_Train, m_Test = split(parts, fold)
        # 预测正确的样本个数
        predict_true = 0
        count(m_Train)
        # 测试样本预测
        for inst in m_Test:
            # inst 为测试集的一个实例
            # print(inst)
            # 真实的类别标签
            true_ClassValue = inst[-1]
            # 创建邻居实例对象
            neighInstances = NeighInstances(K)
            # 从训练集中，找到与 inst 距离最近的 K 个邻居
            for inst_train in m_Train:
                distance = Distance(inst, inst_train)
                neighInstances.add_instance(inst_train[-1], distance)
            # 进行距离加权决策
            probs = [0] * NumClasses
            classCounts = [0] * NumClasses
            weight = [0] * K
            for k in range(0, K):
                weight[k] = 1.0 / (1.0 + neighInstances.get_instances()[k][1] * neighInstances.get_instances()[k][1])
                # 获得第k个邻居的类标索引
                classVlaueIndex = Get_instClassIndex(neighInstances.get_instances()[k][0], info)
                classCounts[classVlaueIndex] = classCounts[classVlaueIndex] + weight[k]
            SUM = sum(weight)
            for k in range(0, NumClasses):
                probs[k] = (classCounts[k] + 1.0) / (SUM + NumClasses)
            # 预测的类标签值的索引
            max_Index = max(enumerate(probs), key=lambda x: x[1])[0]
            # print("真实类标签：{},预测类标签：{}".format(true_ClassValue,info[-1][1][max_Index]))
            if true_ClassValue == info[-1][1][max_Index]:
                # 预测正确
                predict_true = predict_true + 1
        predict = predict_true / len(m_Test)
        predict_List.append(predict)
    #计算十折交叉验证的准确率平均值
    Predict_Avg = sum(predict_List)/10
    print(Predict_Avg)



if __name__ == "__main__":
    # 十次10折交叉验证，取平均
    main()





