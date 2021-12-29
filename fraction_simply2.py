#求分数的最简比（优化版）
x, y = input("输入比0大的分数：\n").split(":")
x, y = int(x), int(y)
min_num = min(x, y)
a = [2]
i = 3
percent = 0.0
print("进度为：{}%".format(percent))
while i <= min_num:
    for each in a:
        if i % each ==0:
            i += 1
            break
    else:
        a.append(i)
        if not percent == round(i/min_num, 3)*100:
            percent = round(i/min_num, 3)*100
            print("进度为：{}%".format(percent))
        while True:
            for j in a:
                if x % j == 0 and y % j ==0:
                    x, y = int(x/j), int(y/j)
                    min_num = min(x, y)
                    break
            else:
                break
        i += 1
print("化简为：{}:{}".format(x, y))