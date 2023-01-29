import pandas as pd
import click
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
df = pd.read_csv('timeline.csv', parse_dates=['Date'])
records = df.to_dict(orient='records')

embeddings = [
    dict(record, embedding=model.encode(record['Description']))
    for record in records
]

def record_for_year(year):
    query_records = [m for m in embeddings if m['Date'].year == year]
    if not query_records:
        return None
    return query_records[0]

def query(year=1963):
    query_record = record_for_year(year)
    query_embedding = model.encode(query_record['Description'])
    scores = [
        (util.dot_score(query_embedding, m['embedding']), m)
        for m in embeddings
    ]
    return sorted(scores, key=lambda pair: pair[0], reverse=True)


@click.command()
@click.argument('year')
def rhymes(year):
    year = int(year)
    click.echo(f"Most significant event of {year}:")
    click.echo("")
    record = record_for_year(year)
    click.echo(record['Description'])

    click.echo("")
    click.echo("Most similar years:")

    nearest = query(year)
    for pair in nearest[1:6]:
        record = pair[1]
        click.echo("")
        click.echo(record['Date'])
        click.echo(record['Description'])


if __name__ == '__main__':
    rhymes()
