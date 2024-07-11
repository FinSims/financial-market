from supabase import create_client, Client
from dotenv import load_dotenv
from Singleton import Singleton


@Singleton
class Supabase:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        self.supabase: Client = create_client(url, key)
