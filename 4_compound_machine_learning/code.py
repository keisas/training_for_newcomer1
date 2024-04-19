from typing import List, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split


def draw_molecule(csvfile: str) -> None:
    # 課題 4-1
    df = pd.read_csv(csvfile)
    smile = df[df["Compound ID"] == "CHEMBL540227"]["SMILES"].iloc[-1]
    mol = Chem.MolFromSmiles(smile)
    label = "CHEMBL540227"
    Draw.MolToFile(mol, f'data/{label}.png')

    return


def create_2d_descriptors(smiles: str) -> Union[npt.NDArray[np.float_], List[float]]:
    # 課題 4-2
    mol = Chem.MolFromSmiles(smiles)
    use_descList = list(filter(lambda p: not p[0] in [
                        "AvgIpc", "SPS"], Descriptors.descList))
    descriptions = [j(mol) for _, j in use_descList]

    return descriptions


def predict_logpapp(csvfile: str) -> Union[npt.NDArray[np.float_], pd.Series, List[float]]:
    # 課題 4-3
    np.random.seed(0)  # 出力を固定するためにseedを指定
    rfr = RandomForestRegressor(random_state=0)  # 出力を固定するためにrandom_stateを指定

    df = pd.read_csv(csvfile)
    use_descList = list(filter(lambda p: not p[0] in [
                        "AvgIpc", "SPS"], Descriptors.descList))
    mols = [Chem.MolFromSmiles(smiles) for smiles in df['SMILES']]
    descriptions = [[j(mol) for _, j in use_descList] for mol in mols]
    descriptions_df = pd.DataFrame(data=descriptions, columns=[
                                   i for i, _ in use_descList])

    X = descriptions_df.values
    y = df['LogP app'].values
    X_train, X_test, y_train, _ = train_test_split(
        X, y, train_size=700, random_state=0)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)

    return y_pred


def grid_search(csvfile: str) -> float:
    # 課題 4-4
    # こちらも出力を固定するためにseedやrandom_stateを指定すること
    np.random.seed(0)
    tuned_parameters = [
        {'n_estimators': [100, 200, 400], 'max_depth': [5, 10, 15]}]
    clf = GridSearchCV(
        RandomForestRegressor(random_state=0),  # 識別器
        tuned_parameters,  # 最適化したいパラメータセット
        cv=4,  # 交差検定の回数
        scoring='neg_mean_squared_error'
    )

    df = pd.read_csv(csvfile)
    use_descList = list(filter(lambda p: not p[0] in [
                        "AvgIpc", "SPS"], Descriptors.descList))
    mols = [Chem.MolFromSmiles(smiles) for smiles in df['SMILES']]
    descriptions = [[j(mol) for _, j in use_descList] for mol in mols]
    descriptions_df = pd.DataFrame(data=descriptions, columns=[
                                   i for i, _ in use_descList])

    X = descriptions_df.values
    y = df['LogP app'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=700, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return rmse


if __name__ == "__main__":
    smiles = "C(=O)(c1ccc(OCCCCCC)cc1)CCNc1cc(Cl)ccc1"
    filepath = "data/fukunishi_data.csv"
    # 課題 4-1
    draw_molecule(filepath)
    # 課題 4-2
    print(create_2d_descriptors(smiles))
    # 課題 4-3
    print(predict_logpapp(filepath))
    # 課題 4-4
    print(grid_search(filepath))
