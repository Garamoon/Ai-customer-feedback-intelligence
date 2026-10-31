"""
Synthetic data generator that mirrors the structure and statistical
patterns of the public 'Women's E-Commerce Clothing Reviews' dataset
(Clothing ID, Age, Title, Review Text, Rating, Recommended IND,
Positive Feedback Count, Division Name, Department Name, Class Name).

NOTE: This environment has no internet access, so the real Kaggle file
can't be downloaded here. This generator produces review text from
templated sentiment/complaint patterns so the full pipeline below is
100% real and runnable end-to-end. To use the real dataset instead,
just drop 'Womens Clothing E-Commerce Reviews.csv' into data/ and
change LOAD_REAL_DATA = True in pipeline.py — the rest of the code
needs zero changes because column names match.
"""
import numpy as np
import pandas as pd
import random
import os

random.seed(42)
np.random.seed(42)

N = 4200

departments = ["Dresses", "Tops", "Bottoms", "Intimates", "Jackets", "Trend"]
division_map = {
    "Dresses": "General", "Tops": "General", "Bottoms": "General",
    "Intimates": "Intimates", "Jackets": "General Petite", "Trend": "Trend"
}
class_map = {
    "Dresses": ["Dresses", "Maxi", "Shirtdresses"],
    "Tops": ["Blouses", "Knits", "Sweaters", "Tees"],
    "Bottoms": ["Pants", "Jeans", "Skirts", "Shorts"],
    "Intimates": ["Lounge", "Swim", "Sleep"],
    "Jackets": ["Jackets", "Outerwear", "Coats"],
    "Trend": ["Trend"]
}

positive_templates = [
    "Absolutely love this {item}! The fit is perfect and the fabric feels amazing.",
    "This {item} exceeded my expectations, great quality for the price.",
    "Beautiful {item}, true to size and so comfortable. Will buy again.",
    "I got so many compliments wearing this {item}. Highly recommend!",
    "The {item} looks even better in person, excellent craftsmanship.",
    "Perfect fit, lovely color, this {item} is now my favorite.",
    "Great value {item}, fits true to size and the material is soft.",
    "This {item} is stylish and comfortable, exactly what I needed.",
]

neutral_templates = [
    "The {item} is okay, nothing special but wearable.",
    "Decent {item} for the price, fit is average.",
    "It's a fine {item}, color is slightly different than pictured.",
    "The {item} is fine, though the fabric feels a bit thin.",
    "Average quality {item}, might keep it, still deciding.",
]

negative_templates_size = [
    "The {item} runs way too small, had to return it.",
    "Sizing is completely off on this {item}, ordered my usual size and it didn't fit.",
    "This {item} was too tight around the shoulders, disappointing fit.",
    "The {item} is oversized, definitely runs large, returning it.",
]
negative_templates_quality = [
    "The fabric of this {item} feels cheap and thin, poor quality.",
    "After one wash the {item} started pilling, low quality material.",
    "The stitching on this {item} was already coming apart out of the box.",
    "This {item} looks nothing like the picture, cheap fabric and bad finish.",
]
negative_templates_shipping = [
    "The {item} arrived late and the packaging was damaged.",
    "Shipping took forever for this {item}, very disappointed with delivery.",
    "Order was delayed and the {item} box was crushed on arrival.",
]
negative_templates_price = [
    "Way overpriced for the quality of this {item}, not worth it.",
    "This {item} is too expensive for such basic material.",
]

complaint_pool = (negative_templates_size + negative_templates_quality +
                   negative_templates_shipping + negative_templates_price)

# generic filler sentences that show up regardless of sentiment, so the
# vocabulary overlaps across classes like real-world review text does
filler_sentences = [
    "Ordered this for a wedding coming up next month.",
    "Bought it in the navy color, true to the online photo.",
    "This was a gift for my sister, she tried it on right away.",
    "I usually shop the petite section for these items.",
    "Wore this to the office and to a weekend brunch.",
    "The store was out of my size so I ordered online instead.",
    "Not my first purchase from this brand, been shopping here for years.",
]

titles_pos = ["Love it!", "Great buy", "So comfy", "Perfect", "Highly recommend", ""]
titles_neg = ["Disappointed", "Not worth it", "Returning it", "Poor quality", "Wrong size", ""]
titles_neu = ["It's okay", "Average", ""]

item_words = {
    "Dresses": "dress", "Tops": "top", "Bottoms": "pair of pants",
    "Intimates": "piece", "Jackets": "jacket", "Trend": "piece"
}

rows = []
for i in range(N):
    dept = random.choice(departments)
    div = division_map[dept]
    cls = random.choice(class_map[dept])
    item = item_words[dept]
    age = int(np.clip(np.random.normal(41, 12), 18, 80))
    clothing_id = random.randint(1000, 1200)

    # rating distribution skewed positive, like the real dataset
    rating = np.random.choice([1, 2, 3, 4, 5], p=[0.06, 0.08, 0.14, 0.30, 0.42])

    if rating >= 4:
        text = random.choice(positive_templates).format(item=item)
        title = random.choice(titles_pos)
        recommended = 1 if random.random() > 0.05 else 0
        # 18% chance of a mild caveat mixed in, like real reviews
        if random.random() < 0.18:
            text += " " + random.choice(neutral_templates + complaint_pool).format(item=item)
    elif rating == 3:
        text = random.choice(neutral_templates).format(item=item)
        title = random.choice(titles_neu)
        recommended = 1 if random.random() > 0.5 else 0
        if random.random() < 0.35:
            text += " " + random.choice(positive_templates + complaint_pool).format(item=item)
    else:
        text = random.choice(complaint_pool).format(item=item)
        title = random.choice(titles_neg)
        recommended = 0 if random.random() > 0.08 else 1
        if random.random() < 0.15:
            text += " " + random.choice(neutral_templates).format(item=item)

    # sprinkle a generic filler sentence into ~40% of reviews (adds shared
    # vocabulary noise across all sentiment classes)
    if random.random() < 0.4:
        text += " " + random.choice(filler_sentences)

    pos_feedback = np.random.poisson(1.2) if rating >= 4 else np.random.poisson(0.4)

    rows.append({
        "Clothing ID": clothing_id,
        "Age": age,
        "Title": title,
        "Review Text": text,
        "Rating": int(rating),
        "Recommended IND": int(recommended),
        "Positive Feedback Count": int(pos_feedback),
        "Division Name": div,
        "Department Name": dept,
        "Class Name": cls,
    })

df = pd.DataFrame(rows)

# inject a few missing values and duplicates on purpose, like real-world data
missing_idx = np.random.choice(df.index, size=40, replace=False)
df.loc[missing_idx[:20], "Review Text"] = np.nan
df.loc[missing_idx[20:], "Division Name"] = np.nan
dup_rows = df.sample(15, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/reviews_raw.csv", index=False)
print("Saved", df.shape, "rows -> data/reviews_raw.csv")
