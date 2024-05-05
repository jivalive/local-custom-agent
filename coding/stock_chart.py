# filename: stock_chart.py
import matplotlib.pyplot as plt

# Replace these placeholders with the actual YTD percentage change data for NVDA and TESLA
nvda_ytd = 0.0
tesla_ytd = 0.0

labels = ['NVDA', 'TESLA']
sizes = [nvda_ytd, tesla_ytd]
colors = ['#7f9cf8', '#00BFFF']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Year-to-Date Stock Price Change')
plt.show()