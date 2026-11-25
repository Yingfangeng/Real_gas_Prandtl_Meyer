from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    seed = 42
    dataset_path = Path(__file__).parent / 'real_gas_PM_dataset'
    datasets = []

    for csv_path in sorted(dataset_path.glob('real_gas_PM_result_*.csv')):
        data = pd.read_csv(csv_path)
        datasets.append(data)

    data = pd.concat(datasets, ignore_index=True)
    data = data[(data['h_2'] != 999) & (data['Q_2'] == -1)]

    features = [
        'T_1', 's_1', 'M_1', 'theta',
        'ns_1', 'Gamma_1', 'P_1', 'rho_1', 'c_1', 'Z_1',
    ]
    X = data[features]
    y = data['M_2']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    print(f'Testing R2 = {r2_score(y_test, prediction):.6f}')

    permutation = permutation_importance(
        model, X_test, y_test,
        scoring='r2', n_repeats=10, random_state=seed,
    )

    importance = pd.DataFrame({
        'variable': features,
        'forest_importance_percent': model.feature_importances_ * 100,
        'permutation_r2_decrease': permutation.importances_mean,
        'permutation_std': permutation.importances_std,
    })
    importance = importance.sort_values('permutation_r2_decrease', ascending=False)

    print(importance.to_string(index=False))
    importance.to_csv(Path(__file__).parent / 'sensitivity_results.csv', index=False)
