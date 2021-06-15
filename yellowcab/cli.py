import click
import pandas as pd
from yellowcab.model import *
import os
from yellowcab.io import input

@click.group()

@click.command()
@click.option('--transform/--not-transform', default=False, help="Provide additional information of position and time")
@click.argument('-i', promt='Please enter input data path', help='Path for input data frame')
@click.argument('-o', prompt='Please enter output file name', help='Path for output')
def transform(transform, i, o):
    if os.path.isdir(i):
        dirname = os.path.dirname(i)
        file = pd.read_parquet(i, engine='pyarraw')
        if transform:
            df = transform_nyc(file)
            df.to_parquet(os.path.join(dirname,f'{o}'))
        else:
            file.to_parquet(os.path.join(dirname,f'{o}'))
        click.echo('Data has been transformed and saved')
    else:
        click.echo('Please provide a valid input data path')

@click.command()
@click.option('--nyc/--queens',help='Chosing based model for prediction')
@click.argument('--predict',prompt='Choose 1 to predict distance, 2 for fare amount, 3 for payment type, 0 for all values', default=False,help='Predict trip distance, fare amount and payment type')
@click.argument('-i', promt='Please enter input data path', help='Path for input data frame')
@click.argument('-o', prompt='Please enter output filename', help='Path for output')
def predict(nyc,predict,i,o):
    if os.path.isdir(i):
        dirname = os.path.dirname(i)
        file = pd.read_parquet(i, engine='pyarraw')
        if nyc:
            if predict == 0:
                model = model_nyc().predict(file)
            elif predict == 1:
                model = model_nyc().predict_distance_nyc(file)
            elif predict == 2:
                model = model_nyc().predict_fare_nyc(file)
            else:
                model = model_nyc().predict_payment_type_nyc(file)
        else:
            if predict == 0:
                model = model_queens().predict(file)
            elif predict == 1:
                model = model_queens().predict_distance_queens(file)
            elif predict == 2:
                model = model_queens().predict_fare_queens(file)
            else:
                model = model_queens().predict_payment_type_queens(file)
            file.to_parquet(os.path.join(dirname, f'{o}'))
        click.echo('Predicted data has been saved')
        click.echo('Predicted values are', model)
    else:
        click.echo('Please provide a valid input data path')


if __name__ == '__main__':
    transform()
    predict()

