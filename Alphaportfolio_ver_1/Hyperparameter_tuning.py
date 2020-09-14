from main import main
import pandas as pd
from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns

#Analysis parameter Set_up
min_epoch_size = 500
max_epoch_size = 5000
step = 500
what_to_analyze = 'Sharpe_Ratio' #Cumulative_Return


def test_result_analysis(train_data_path, test_data_path, what_to_analyze):
    train_data = pd.read_csv(train_data_path)
    train_data.set_index('Investment Period', inplace=True)
    test_data = pd.read_csv(test_data_path)
    test_data.set_index('Investment Period', inplace=True)
    train_data = train_data.loc[:, train_data.columns.str.startswith(f'{what_to_analyze}')]

    def last_value(row_data):
        # value = row_data.index
        value = row_data[row_data.last_valid_index()]
        return value

    train_last_result = train_data.apply(lambda x: last_value(x), axis=1)
    train_last_result = train_last_result.mean()
    test_last_result = test_data.apply(lambda x: last_value(x), axis=1)
    test_last_result = test_last_result.mean()

    return train_last_result, test_last_result


#Run Analysis

result_df = pd.DataFrame()

for epoch_num in range(min_epoch_size,max_epoch_size+1,step):

    main(epoch_num)
    sleep(2)
    train_result, test_result = test_result_analysis('./test_log_reward.csv', './train_log_reward.csv', 'Sharpe_Ratio')

    result_df.at[f'{epoch_num}', 'Train'] = train_result
    result_df.at[f'{epoch_num}', 'Test'] = test_result

# Name index of test_log
result_df.index.name = 'Epoch_Number'
result_df.reset_index(inplace=True)


#Draw a result figure

plt.figure(figsize=(12,6))
ax = sns.lineplot(data=result_df, x = 'Epoch_Number', y = 'Train',
                  label = 'Train_Set', color='b', linewidth=2.5, sort=False)
ax = sns.lineplot(data=result_df, x = 'Epoch_Number', y = 'Test',
                  label = 'Test_Set', color='r', linewidth=2.5, sort=False)
ax.set(xlabel='Epoch_Number', ylabel=f'Avg {what_to_analyze}')
plt.legend()
plt.title(f'Average {what_to_analyze} Comparison',fontsize = 20)
plt.savefig(f'Average_{what_to_analyze}_Comparison.jpg')






