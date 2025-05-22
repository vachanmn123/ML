import pandas as pd
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

df = pd.read_csv('C:/Users/bieta/OneDrive/Desktop/BigBasket.csv')

user_data = {
    'user_id': [1, 2, 3],
    'favorite_category': [
        'Beverages',
        'Snacks & Branded Foods',
        'Fruits & Vegetables'
    ]
}

user_df = pd.DataFrame(user_data)

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_ai_ad(
    user_id,
    favorite_category,
    product_name,
    brand,
    discount_price,
    original_price,
    url
):
    prompt = (
        f"User {user_id} loves {favorite_category}. "
        f"Generate a creative and engaging ad for {product_name} by {brand}, "
        f"which is available at a discounted price of ₹{discount_price} "
        f"(original price: ₹{original_price}). "
        f"Encourage the user to click on the link: {url}."
    )
    inputs = tokenizer.encode(
        prompt, return_tensors="pt", max_length=512, truncation=True
    )
    outputs = model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7
    )
    ad_message = tokenizer.decode(
        outputs[0], skip_special_tokens=True
    )
    return ad_message

def generate_ad(user_id, favorite_category, products):
    category_products = products[products['Category'] == favorite_category]
    if not category_products.empty:
        product = category_products.sample(1).iloc[0]
        ad_message = generate_ai_ad(
            user_id,
            favorite_category,
            product['ProductName'],
            product['Brand'],
            product['DiscountPrice'],
            product['Price'],
            product['Absolute_Url']
        )
    else:
        ad_message = (
            f"Hey User {user_id}! We couldn't find any products in your favorite "
            f"category ({favorite_category}). Check out our other offerings!"
        )
    return ad_message

user_df['ad_message'] = user_df.apply(
    lambda row: generate_ad(row['user_id'], row['favorite_category'], df), axis=1
)

print("\nAI-Generated Ad Messages:")
for _, row in user_df.iterrows():
    print(row['ad_message'])
