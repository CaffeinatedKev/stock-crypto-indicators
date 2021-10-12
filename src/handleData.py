import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import json

file = open('bitcoin-data.json')
data = json.load(file)

date = []
x = []
iteration = 1.34
y = []
for i in data:
    date.append(mdates.datestr2num(i['Date']))
    x.append(iteration)
    y.append(i['Close'])
    iteration += 1

log_x = np.log(x)
log_y = np.log(y)

coefficients = np.polyfit(log_x, y, 1)

c = coefficients[0] * log_x - coefficients[1]
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=300))
plt.plot(date, log_y)
plt.gcf().autofmt_xdate()
# plt.plot(date, c)
plt.show()

