#from collections import 
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def deserialize(data):
    data_list = data[1:-1]
    data_list = data_list.split(',')
    
    
    if len(data_list) == 1:
        return None if data_list[0] == "null" else data_list[0]
    
    root = TreeNode(data_list.pop(0))
    q = [root]

    while q:
        parent = q.pop(0)
        if parent:
            left = data_list.pop(0)
            right = data_list.pop(0)
            parent.left = TreeNode(int(left)) if left != "null" else None
            parent.right = TreeNode(int(right)) if right != "null" else None
            q.append(parent.left)
            q.append(parent.right)
    
    return root


data = ['1','2','3','null','null','4','5']
data = "[" + ','.join(data) + "]"
print(data)

deserialize(data)