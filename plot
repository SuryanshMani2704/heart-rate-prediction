
sns.countplot(x='age', data=df)
plt.title('Distribution of Heart Attack')
plt.show()

sns.pairplot(df)
plt.show()

plt.figure(figsize=(14,12))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

X = df.drop('oldpeak', axis=1)
y = df['thal']
from sklearn.model_selection import train_test_split


X = df.drop(columns=['oldpeak'])
y = df['thal']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

