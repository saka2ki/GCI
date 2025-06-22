import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def target_rate(df_train, df_test, numerical_feature, boolean_feature, target, numerical, categorical, dum_categorical, q=True, n=10, is_annot=True):
    if q:
        df_train['qcut'] = pd.qcut(df_train[numerical_feature], n, duplicates='drop')
    else:
        df_train['qcut'] = pd.cut(df_train[numerical_feature], n, duplicates='drop')
    target_rate = df_train.groupby([boolean_feature, 'qcut'], observed=False)[target].mean().reset_index()
    target_rate = df_train.groupby(['qcut', boolean_feature], observed=False)[target].mean().reset_index()
    
    # 死亡率と件数を集計
    summary = df_train.groupby(['qcut', boolean_feature], observed=False).agg(
        target_rate=(target, 'mean'),
        count=(target, 'count')
    ).reset_index()
    
    # ピボットテーブル作成（数値用とラベル用）
    heatmap_values = summary.pivot(index='qcut', columns=boolean_feature, values='target_rate')
    heatmap_counts = summary.pivot(index='qcut', columns=boolean_feature, values='count')
    
    # アノテーション用の文字列（例: "0.75\n(n=12)"）
    annot = heatmap_values.copy().astype("object")
    for row in annot.index:
        for col in annot.columns:
            val = heatmap_values.loc[row, col]
            cnt = heatmap_counts.loc[row, col]
            annot.loc[row, col] = f"{val:.2f}(n={cnt})"
    
    f, ax = plt.subplots(1, 2, figsize=(16, 6), facecolor='gray')
    sns.heatmap(
        heatmap_values.astype(float),  # 数値
        annot=annot,                  # アノテーション
        fmt='',                       # アノテーションに文字列を使う
        cmap='YlGnBu',
        cbar_kws={'label': f'{target} Rate'},
        ax=ax[0]
    )
    ax[0].set_title(f'{target} Rate by Age Group ({numerical_feature}) and {boolean_feature} with Sample Size')
    ax[0].set_xlabel(boolean_feature)
    ax[0].set_ylabel(numerical_feature)
    
    sns.heatmap(df_train[[*numerical, *dum_categorical, target]].corr(),annot=is_annot,cmap='bwr',linewidths=0.2, ax=ax[1])
    plt.tight_layout()
    #plt.show()
######################################################################################################################################################################
    length = len(numerical + categorical)
    f, ax = plt.subplots(3, length, figsize=(4*length, 10), facecolor='gray', squeeze=False)
    i = 0
    
    for col in numerical + categorical:
        if col in numerical:
            # 数値列: 分布 + クラスごとの分布
            sns.histplot(df_train[col], kde=True, ax=ax[0, i], color="#1f77b4")
            ax[0, i].set_title(f'Distribution of {col}')
            ax[0, i].set_ylabel('Train')
    
            sns.histplot(data=df_train, x=col, hue=target, kde=True, ax=ax[1, i])
            ax[1, i].set_title(f'{col}: Distribution by Perished')
            ax[1, i].set_ylabel('Train')
    
            sns.histplot(df_test[col], kde=True, ax=ax[2, i], color="#1f77b4")
            ax[2, i].set_title(f'Distribution of {col}')
            ax[2, i].set_ylabel('Test')
    
            i += 1
    
        else:
            # カテゴリ列: 棒グラフ + クラスごとのカウント
            df_train[col].value_counts().plot.bar(
                color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax=ax[0, i]
            )
            ax[0, i].set_title(f'Number of Passengers by {col}')
            ax[0, i].set_ylabel('Train')
    
            sns.countplot(x=col, hue=target, data=df_train, ax=ax[1, i])
            ax[1, i].set_title(f'{col}: Perished vs Survived')
            ax[1, i].set_ylabel('Train')
    
            df_test[col].value_counts().plot.bar(
                color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax=ax[2, i]
            )
            ax[2, i].set_title(f'Number of Passengers by {col}')
            ax[2, i].set_ylabel('Test')
    
            i += 1
        #break
    plt.tight_layout()
    plt.show()
