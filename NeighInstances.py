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

# # 使用示例
# K = 5  # 指定邻居实例集合的大小
# neigh_instances = NeighInstances(K)

# 添加实例
# neigh_instances.add_instance('class1', 0.1)
# neigh_instances.add_instance('class2', 0.3)
# neigh_instances.add_instance('class3', 0.5)
# neigh_instances.add_instance('class4', 0.2)
# neigh_instances.add_instance('class5', 0.4)
# neigh_instances.add_instance('class6', 0.15)  # 这个实例会替换掉最远的邻居

# 获取并打印当前的邻居实例集合
# print(neigh_instances.get_instances())
