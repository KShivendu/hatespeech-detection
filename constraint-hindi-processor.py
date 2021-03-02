import pandas as pd

df = pd.read_csv('data/constraint_Hindi_Train-Sheet1.csv')
df = df.drop(['Unique ID'], axis=1)
df['Labels Set'] = [(1 if 'hate' in labels else 0)
                    for labels in df['Labels Set']]
df.columns = ['comment', 'isHate']
print(df['isHate'][:10])
print(f"{sum(df['isHate'])} (label : hate) + {len(df) - sum(df['isHate'])} (label : not-hate) = {len(df)}")


df.to_csv('data/constraint_Hindi_Train-cleaned.csv', index=False)
