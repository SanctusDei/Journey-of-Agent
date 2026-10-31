def create_resource():
    print("创建资源")
    return object()

print("开始定义函数")

def retrieve(resource = create_resource()):
    return resource

print("函数定义完成")
first = retrieve()
second = retrieve()
print("两次是否使用同一对象：", first is second)