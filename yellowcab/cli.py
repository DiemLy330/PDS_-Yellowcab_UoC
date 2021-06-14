import click
from yellowcab.cabana import trips_info
from . import io
from . import cabana
from . import model


@click.command()
@click.option('--transform/--no-transform', default=False, help="Transform dataset into the augmented format and "
                                                                "save to disk")
@click.option('-i', type=click.File('rb'), default=None, help="Input file")
@click.option('-o', type=click.File('wb'), default=None, help="Output file")
@click.option('--predict/--no-predict', default=False)
def main(transform, i, o, predict):
    if transform:
        cabana.trips_info(i).get_duration()
        cabana.trips_info(i).get_time()
        cabana.trips_info(i).get_position()

        if i is None or o is None:
            print("Insufficient file specification, use -i for input path and -o for output path")
        else:
            print("Transforming data ...")

    elif predict:
        io.input.read_model()
        print("Predicting...")


if __name__ == '__main__':
    main()
