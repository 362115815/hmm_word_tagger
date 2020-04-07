import pandas as pd
import json
df = pd.read_csv('brown.csv')

with open('brown_map.json','r') as f:

    uni_tag = json.load(f)

df['universal_pos'] = ''

for i in range(len(df)):
    a = df.iloc[i, 4]
    b = df.iloc[i, 5]
    a = a.split(' ')
    b = b.split(' ')
    uni_pos = []
    for j, item in enumerate(b):
        try:
            uni_t = uni_tag[item.upper()]
            uni_pos.append(uni_t)

        except Exception as e:
            raw_text = df.iloc[i, 3]
            raw_text = raw_text.split(' ')[j].rsplit('/', 1)
            true_text = raw_text[0]
            true_tag = raw_text[1]
            b[j] = true_tag
            a[j] = true_text
            df.iloc[i, 4] = ' '.join(a)
            df.iloc[i, 5] = ' '.join(b)

    df.iloc[i, -1] = ' '.join(uni_pos)

df.to_csv('brown.csv', index = False)