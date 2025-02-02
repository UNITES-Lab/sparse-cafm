# proof i can level-order traverse a tree
# max efficiency solution below lol

class Node:
    def __init__(self, val: int, child_left, child_right) -> None:
        self.val = val
        self.child_left = child_left
        self.child_right = child_right

n6 = Node(6, None, None)
n5 = Node(5, None, None)
n4 = Node(4, None, None)
n3 = Node(3, n5, n6)   
n2 = Node(2, n4, None)   
n1 = Node(1, n2, n3)

queue = {}

def level_order_traversal(node: Node, level: int):
    # 1. iterate through tree, mark level of recursion
    if node == None:
        return None, None
    level_order_traversal(node.child_left, level + 1)
    level_order_traversal(node.child_right, level + 1)
    if level not in queue:
        queue[level] = [node.val]
    else:
        queue[level].append(node.val)
    
level_order_traversal(n1, 0)
for k in sorted(list(queue.keys())):
    for val in queue[k]:
        print(val, end = " ")
    print()