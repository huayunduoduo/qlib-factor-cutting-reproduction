"""Score normalized news with a Chinese financial sentiment model.

Default model: bardsai/finance-sentiment-zh-base. It was trained on a Chinese
translation of Financial PhraseBank and is NOT guaranteed to fit A-share news;
manually inspect positive/negative/neutral samples before research use. Model
training data may overlap history, so text leakage cannot be ruled out.

Install dependencies first: python -m pip install -r requirements-news.txt
The first run downloads the model and needs network access.
"""
import argparse
from pathlib import Path

import pandas as pd

MODEL = 'bardsai/finance-sentiment-zh-base'
SCORE = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('--out', default='data/news_raw/news_with_scores.csv')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--model', default=MODEL)
    p.add_argument('--batch-size', type=int, default=16)
    a = p.parse_args()

    try:
        from transformers import pipeline
    except ImportError:
        raise SystemExit('Missing dependency. Run: '
                         'python -m pip install -r requirements-news.txt')

    df = pd.read_csv(a.csv)
    if a.limit:
        df = df.head(a.limit).copy()
    texts = (df['title'].fillna('') + '。' + df['content'].fillna('')).str.slice(0, 500)

    clf = pipeline('sentiment-analysis', model=a.model, tokenizer=a.model,
                   truncation=True, max_length=512)
    labels, probs, sentiments = [], [], []
    for start in range(0, len(texts), a.batch_size):
        results = clf(list(texts.iloc[start:start + a.batch_size]))
        for r in results:
            label = str(r['label']).lower()
            labels.append(label)
            probs.append(float(r['score']))
            sentiments.append(SCORE.get(label, 0.0) * float(r['score']))
        print('scored %d/%d' % (min(start + a.batch_size, len(texts)), len(texts)))

    df['sentiment_label'] = labels
    df['sentiment_confidence'] = probs
    df['sentiment'] = sentiments  # signed, weighted by confidence, within [-1, 1]
    if not df['sentiment'].between(-1, 1).all():
        raise SystemExit('Sentiment outside [-1,1]; inspect the model output mapping.')

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding='utf-8')
    print('Wrote %d scored rows -> %s' % (len(df), out))
    print('Reminder: spot-check samples by hand before trusting these scores.')


if __name__ == '__main__':
    main()
