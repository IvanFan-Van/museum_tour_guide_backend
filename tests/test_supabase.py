import os
import traceback
from supabase import create_client
from dotenv import load_dotenv
from postgrest.exceptions import APIError

load_dotenv()

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]

try:
    supabase = create_client(url, key)
except Exception as e:
    print(f"failed to create client: {e}")
    traceback.print_exc()
    exit(1)

tables = ["sections", "images", "paragraphs"]
for table_name in tables:
    try:
        response = supabase.table(table_name).select("*", count="exact").limit(0).execute()
        count = response.count
        print(f"Table {table_name}: EXISTS | Rows: {count}")
    except APIError as e:
        if "42P01" in str(e) or "not found" in str(e).lower():
            print(f"Table {table_name}: DOES NOT EXIST")
        else:
            print(f"Table {table_name}: ERROR OCCURRED {e}")
    except Exception as e:
        print(f"Table {table_name}: Unexpected Error {e}")
