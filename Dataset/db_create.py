import pandas as pd
import sqlite3 as sql

DataBasePath="./database.sqlite"
conn=sql.connect(DataBasePath)

filePath="./dataset/"    
datafiles=["olist_customers_dataset","olist_geolocation_dataset","olist_order_items_dataset","olist_order_payments_dataset","olist_order_reviews_dataset","olist_orders_dataset","olist_products_dataset","olist_sellers_dataset","product_category_name_translation"]

for f in datafiles:
    df=pd.read_csv(filePath+f+".csv")        
    df.to_sql(name=f,con=conn,if_exists='append', index=False)
conn.close()
