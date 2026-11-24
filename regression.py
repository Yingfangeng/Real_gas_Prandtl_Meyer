from pathlib import Path
import random
import pandas as pd
from pysr import PySRRegressor
from sklearn.metrics import r2_score


if __name__ == '__main__':
    
    seed = 42
    dataset_path = Path(__file__).parent / 'real_gas_PM_dataset'
    datasets = []

    for csv_path in sorted(dataset_path.glob('real_gas_PM_result_*.csv')):
        data = pd.read_csv(csv_path)
        datasets.append(data)

    data = pd.concat(datasets, ignore_index=True)
    data = data[data['h_2'] != 999]

    indices = list(range(len(data)))
    random.Random(seed).shuffle(indices)
    data = data.iloc[indices].reset_index(drop=True)

    # Dataset split
    training_end = int(0.8 * len(data))
    validation_end = int(0.9 * len(data))
    training = data.iloc[:training_end]
    validation = data.iloc[training_end:validation_end]
    testing = data.iloc[validation_end:]

    splits = {'training': training, 'validation': validation, 'testing': testing}
    
    for name, split in splits.items():
        split.to_csv(dataset_path / f'{name}.csv', index=False)

    features = ['s_1', 'T_1', 'M_1', 'theta']

    model = PySRRegressor(
        niterations=100,
        maxsize=15,
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['square', 'sqrt'],
        elementwise_loss='loss(prediction, target) = ((prediction - target) / target)^2',
        model_selection='best',
        precision=64,
    )


    model.fit(training[features], training['h_2'])

    # Determine which formula to use according to the validation set
    best_loss = float('inf')
    best_index = 0

    for index in range(len(model.equations_)):
        prediction = model.predict(validation[features], index=index)
        target = validation['h_2'].to_numpy()
        loss = (((prediction - target) / target) ** 2).mean()
        
        if loss < best_loss:
            best_loss = loss
            best_index = index

    formula = str(model.sympy(index=best_index))
    print('h_2 =', formula)

    final_result = {
        'formula': formula,
        'complexity': model.equations_.iloc[best_index]['complexity']}

    for name, split in splits.items():

        prediction = model.predict(split[features], index=best_index)
        target = split['h_2'].to_numpy()
        loss = (((prediction - target) / target) ** 2).mean()
        r2 = r2_score(split['h_2'], prediction)
        rmse = (((prediction - target) ** 2).mean()) ** 0.5
        final_result[f'{name}_r2'] = r2
        final_result[f'{name}_rmse'] = rmse
        final_result[f'{name}_relative_mse'] = loss
        print(f'{name}: R2 = {r2:.6f}, RMSE = {rmse:.6g}, relative MSE loss = {loss:.6g}')

    output_path = Path(model.output_directory_) / model.run_id_
    pd.DataFrame([final_result]).to_csv(output_path / 'final_formula.csv', index=False)

    for pattern in ['*.pkl', '*.bak']:
        for file_path in output_path.glob(pattern):
            file_path.unlink()