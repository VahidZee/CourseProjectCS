class Node:
    def __init__(self, p, next=None, prev=None):
        self.next = next
        self.prev = prev
        self.patient = p

    def __repr__(self):
        result = ''
        node = self
        while node is not None:
            result = f'{result} - {node.patient.id}'
            node = node.next
        return result

    def count(self):
        count = 0
        node = self
        while node is not None:
            count += 1
            node = node.next
        return count
