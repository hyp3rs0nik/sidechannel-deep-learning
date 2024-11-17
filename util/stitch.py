import pandas as pd

def adjust_timestamps(file_path, gap_threshold=1800000):
    
    df = pd.read_csv(file_path)
    df['timestamp_diff'] = df['timestamp'].diff()
    
    offset = 0
    for i in range(1, len(df)):
        if df.loc[i, 'timestamp_diff'] > gap_threshold:
            
            offset += df.loc[i, 'timestamp_diff'] - 1  
        
        df.loc[i, 'timestamp'] -= offset

    df.drop(columns=['timestamp_diff'], inplace=True)
    return df


def continue_sequence_index(df):
    reset_start_index = df[df['sequenceIndex'] == 0].index[0]
    if reset_start_index == 0:
        previous_sequence_index = df['sequenceIndex'].max()  
    else:
        previous_sequence_index = df.loc[reset_start_index - 1, 'sequenceIndex']
    
    current_sequence_index = 0
    for i in range(reset_start_index, len(df)):
        if df.loc[i, 'sequenceIndex'] != current_sequence_index:
            previous_sequence_index += 1
            current_sequence_index = df.loc[i, 'sequenceIndex']
        
        df.loc[i, 'sequenceIndex'] = previous_sequence_index

    return df

adjust_timestamps('./data/training/v3.sensors.csv').to_csv('./data/training/clean_v3.sensors.csv', index=False)
continue_sequence_index(adjust_timestamps('./data/training/v3.keystrokes.csv')).to_csv('./data/training/clean_v3.keystokes.csv', index=False)
