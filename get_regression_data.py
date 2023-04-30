import click
from sklearn.datasets import fetch_california_housing, fetch_openml, load_diabetes

DATA_REPO = "data/"


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["california", "boston", "diabetes"]),
    default="california",
    help="Choose a dataset to download.",
)
def main(dataset):
    if dataset == "california":
        data = fetch_california_housing(as_frame=True)
    elif dataset == "boston":
        data = fetch_openml(data_id=531, as_frame=True)
    elif dataset == "diabetes":
        data = load_diabetes(as_frame=True)
    else:
        click.echo(
            "Invalid dataset name. Please choose from 'california_housing', 'boston_housing', or 'diabetes'."
        )
        return
    save_path = f"{DATA_REPO}{dataset}.pkl"
    data.frame.to_pickle(save_path)
    click.echo(
        f"Downloaded {dataset} dataset with shape: {data.frame.shape}\nData stored in {save_path}"
    )


if __name__ == "__main__":
    main()
