from openai.types.chat import ChatCompletionMessageParam

COLUMN_PREDICTION_TYPES = ["real", "integer", "text", "date", "datetime"]

COLUMN_PREDICTION_PROMPT: list[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": f"""Given a SQL database and question, please determine a list of "Output Types" and "Input Columns" required to answer the question.
Before each list item, write a `--` line comment explaining why it is needed, citing the user's question when possible.
Output types should be one of: {", ".join(COLUMN_PREDICTION_TYPES)}
Input columns should be formatted without quotes as: table_name.column_name
Ensure that the output types provide exactly the information needed to answer the question and nothing more or less.""",
    },
    {
        "role": "user",
        "content": """Given the database:
<schema>
# Table: sales
Foreign key: (product_id) references products (product_id) many-to-one
Columns:
sales.sale_date\tDATE
sales.product_id\tINTEGER
sales.quantity\tREAL
# Table: prices
Primary key: (price_date, product_id)
Foreign key: (product_id) references products (product_id) many-to-one
Columns:
prices.price_date\tDATE
prices.product_id\tINTEGER
prices.price\tREAL\tPrice for product_id on price_date
# Table: products
Primary key: (product_id)
Columns:
products.product_id\tINTEGER
products.product_name\tTEXT
</schema>
What was the average price by product name in January 2024?""",
    },
    {
        "role": "assistant",
        "content": """Output Types
-- The 'product name' requested by the question
text
-- The average price 'in January 2024' for each product name
real
Input Columns
-- The question asks for the 'average price'. We should aggregate the price column in prices
prices.price
-- To filter for prices 'in January 2024', we should use the price_date column in prices
prices.price_date
-- We need to join prices with products by product_id to obtain the product name
prices.product_id
-- Join key for product_id in prices
products.product_id
-- Finally, we can group by the product_name column in products
products.product_name""",
    },
]
