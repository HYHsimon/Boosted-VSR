# x = input()
# y = input()
#
# x1 = x.split(" ")[0]
# x2 = int(x.split(' ')[1])
#
#
# def is_symmetrical(str):
#     length = len(str)
#     count =  0
#     for i in range(length // 2):
#         if str[i] == str[length - i - 1]:
#             count = count + 1
#     return count
#
# num = is_symmetrical(y)
#
# y1 = y[num:]
#
# for i in range(x2 - 1):
#     y += y1
#
# print(y)

'''

5 10
3 9 5 7 6
'''


x = input()
y = input()

x1 = x.split(" ")[0]
x2 = int(x.split(' ')[1])

y = sorted([int(xx) for xx in y.split(' ')])
s = 0
count = 0

for i in y:
    if s < x2:
        s += i
    if s <= x2:
        count =count +1


print(count)


