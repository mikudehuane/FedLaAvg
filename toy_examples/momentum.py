from matplotlib import pyplot as plt

# test momentum feasibility
E1 = 10
E2 = 40
ratio1 = 0.0
ratio2 = 0.0
ratio1_his = []
ratio2_his = []
r1_sum = 0.0
r2_sum = 0.0

gamma = 0.1

it = 0
for _ in range(1000):
    for _ in range(E1):
        ratio1_his.append(ratio1)
        ratio2_his.append(ratio2)
        ratio1 = ratio1 * gamma + (1-gamma)
        ratio2 = ratio2 * gamma
        it += 1
        r1_sum += ratio1
        r2_sum += ratio2
        print(f"w1: {r1_sum / it}, w2: {r2_sum / it}")
    for _ in range(E2):
        ratio1_his.append(ratio1)
        ratio2_his.append(ratio2)
        ratio1 = ratio1 * gamma
        ratio2 = ratio2 * gamma + (1-gamma)
        it += 1
        r1_sum += ratio1
        r2_sum += ratio2
        print(f"w1: {r1_sum / it}, w2: {r2_sum / it}")
ratio1_his.append(ratio1)
ratio2_his.append(ratio2)
# plt.plot(range(len(ratio1_his)), ratio1_his, c='red')
# plt.plot(range(len(ratio2_his)), ratio2_his, c='blue')
# plt.show()
