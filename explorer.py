import matplotlib.pyplot as plt
import seaborn as sns

def plot_price_distribution(data):
    plt.figure(figsize=(12, 6))
    sns.histplot(data, kde=True, bins=30)
    plt.title('南瓜价格分布')
    plt.savefig('price_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_price_by_variety(data):
    top_varieties = data['Variety'].value_counts().nlargest(5).index
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='Variety', y='Avg_Price', data=data[data['Variety'].isin(top_varieties)])
    plt.title('前5品种的价格分布')
    plt.savefig('price_by_variety.png', bbox_inches='tight', dpi=300)
    plt.close()