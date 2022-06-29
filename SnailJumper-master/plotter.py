import matplotlib.pyplot as plt
file = open("data.txt", "r")

maxes = []
means = []
mins = []

for line in file:
    best, worst, mean = line.split()
    maxes.append(float(best))
    means.append(float(mean))
    mins.append(float(worst))

plt.plot(maxes, label="Max Fitnesses")
plt.plot(means, label="Mean Fitnesses")
plt.plot(mins, label="Min Fitnesses")
plt.show()
