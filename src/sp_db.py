import os
import logging
from pathlib import Path
from supabase import create_client, Client
from dotenv import load_dotenv


os.chdir(Path(__file__).parent)
load_dotenv()

logger = logging.getLogger()

SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_KEY"]


def get_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def insert_history(client, best_acc, best_val_acc, min_loss, min_val_loss, epochs_count):
    data = {
        "accuracy": best_acc,
        "val_accuracy": best_val_acc,
        "loss": min_loss,
        "val_loss": min_val_loss,
        "epochs_count": epochs_count
    }
    #rows = [{"content": note} for note in data ]

    response = client.table("training_history").insert(data).execute()
    return response


def fetch_history(client):
    """Loads data from Supabase for display in a table."""
    try:
        response = client.table("training_history").select("*").order("accuracy", desc=True).execute()
        return response.data

    except Exception as e:
        logging.error(f"Error fetching history: {e}")
        return []


# def clear_all_history(client):
#     """Deletes all records from the training_history table."""
#     # In Supabase, you can use the .neq("id", 0) filter to delete all rows since ID is always greater than 0.
#     response = client.table("training_history").delete().neq("id", 0).execute()
#     return response


def main(history_dict):
    if not history_dict:
        logging.warning("No history dictionary provided.")
        return

    try:
        # 0. Extract the required 4 values from the history
        best_acc = float(max(history_dict['accuracy']))
        best_val_acc = float(max(history_dict['val_accuracy']))
        min_loss = float(min(history_dict['loss']))
        min_val_loss = float(min(history_dict['val_loss']))

        # The number of epochs is simply the length of the list in history
        epochs_count = int(len(history_dict['accuracy']))

        # 1. Create a Connection Client
        supabase = get_client()

        # 2. Insert the Data to supabase
        insert_history(supabase, best_acc, best_val_acc, min_loss, min_val_loss, epochs_count)
        logging.info(f"Successfully saved metrics for {epochs_count} to Supabase")

    except KeyError as e:
        logging.error(f"Missing key in history dictionary: {e}")
    except Exception as e:
        logging.error(f"Error during Supabase process: {e}")

    #clear_all_history(client)


if __name__ == "__main__":

    main(history_dict)
